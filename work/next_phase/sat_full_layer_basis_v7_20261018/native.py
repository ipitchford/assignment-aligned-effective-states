"""One bounded extraction of the complete final MLP state and exact readout."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import traceback

import numpy as np
import torch

from protocol import *


def save_npz_exclusive(path, **arrays):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def save_json_exclusive(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_runtime():
    spec = importlib.util.spec_from_file_location("sat_full_layer_pinned_runtime", RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct pinned runtime import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer(model, rows, hook_points):
    rows = np.asarray(rows, dtype=np.int64)
    cache = {}; hook_counts = {}
    def capture(key):
        def hook(tensor, name=None, hook=None):
            hook_counts[key] = hook_counts.get(key, 0) + 1
            cache[key] = tensor[:, -1, :].detach().cpu().numpy().copy()
            return None
        return hook
    for key, hook_point in hook_points.items():
        hook_point.add_hook(capture(key))
    try:
        with torch.inference_mode():
            cache["logits"] = model(torch.from_numpy(rows))[:, -1, :].detach().cpu().numpy().copy()
    finally:
        model.remove_all_hooks()
    expected = {"pre": (len(rows), 512), "post": (len(rows), 512),
                "resid_mid": (len(rows), 128), "resid_post": (len(rows), 128),
                "logits": (len(rows), 16)}
    if set(cache) != set(expected):
        raise RuntimeError(f"hook inventory mismatch: {sorted(cache)}")
    if hook_counts != {key: 1 for key in ("pre", "post", "resid_mid", "resid_post")}:
        raise RuntimeError(f"hook firing-count mismatch: {hook_counts}")
    for key, shape in expected.items():
        if cache[key].shape != shape:
            raise RuntimeError(f"hook shape mismatch for {key}: {cache[key].shape}")
    if not np.array_equal(cache["post"], np.maximum(cache["pre"], 0)):
        raise RuntimeError("exact ReLU identity failed")
    if not all(value.dtype == np.float32 and np.isfinite(value).all() for value in cache.values()):
        raise RuntimeError("native dtype or finiteness failure")
    return cache


def extract(model, tokens, ids, replay, hook_points):
    keys = ("pre", "post", "resid_mid", "resid_post", "logits")
    if not replay:
        gathered = {key: [] for key in keys}
        for start in range(0, len(ids), NATIVE_BATCH_SIZE):
            values = infer(model, tokens[ids[start:start + NATIVE_BATCH_SIZE]], hook_points)
            for key in keys: gathered[key].append(values[key])
        return {key: np.concatenate(gathered[key], axis=0) for key in keys}, int(len(ids)), int((len(ids) + NATIVE_BATCH_SIZE - 1) // NATIVE_BATCH_SIZE)
    starts = sorted(set((int(uid) // NATIVE_BATCH_SIZE) * NATIVE_BATCH_SIZE for uid in ids))
    by_id = {}
    input_forwards = 0
    for start in starts:
        block_ids = np.arange(start, min(start + NATIVE_BATCH_SIZE, len(tokens)), dtype=np.int32)
        values = infer(model, tokens[block_ids], hook_points)
        input_forwards += len(block_ids)
        for uid in ids:
            if start <= int(uid) < start + len(block_ids):
                by_id[int(uid)] = {key: values[key][int(uid) - start].copy() for key in keys}
    if set(by_id) != set(map(int, ids)):
        raise RuntimeError("batch-matched replay did not cover every fixed id")
    arrays = {key: np.stack([by_id[int(uid)][key] for uid in ids]) for key in keys}
    return arrays, input_forwards, len(starts)


def execute(args, frozen, versions):
    preflight = verify_live_preflight()
    with np.load(BANK, allow_pickle=False) as bank:
        tokens = bank["unique_tokens"].copy()
    if tokens.shape != (EXPECTED_ROWS, 41):
        raise RuntimeError("bank token shape mismatch")
    construction = json.loads(PRIOR_CONSTRUCTION.read_text())
    with np.load(PRIOR_NATIVE, allow_pickle=False) as prior:
        prior_arrays = {key: prior[key].copy() for key in ("pre", "post", "logits", "unique_ids")}
    ids = np.arange(len(tokens), dtype=np.int32)
    replay = args.stage == "replay"
    if replay:
        ids = np.asarray(construction["replay_unique_ids"], dtype=np.int32)
        if ids.shape != (8,) or len(np.unique(ids)) != 8 or np.any(ids < 0) or np.any(ids >= len(tokens)):
            raise RuntimeError("invalid replay ids")
    output_path = HERE / ("full_native_outputs.npz" if not replay else "replay_outputs.npz")
    receipt_path = HERE / ("native_receipt.json" if not replay else "replay_receipt.json")
    if output_path.exists() or receipt_path.exists():
        raise RuntimeError("exclusive output already exists")
    forward_marker = HERE / f"forward_started_{args.stage}.json"
    if forward_marker.exists():
        raise RuntimeError("forward-start marker already exists")
    runtime = load_runtime()
    model, config, model_receipt = runtime.load_model(threads=NATIVE_THREADS)
    if len(model.blocks) != 2 or model.blocks[-1] is not model.blocks[1]:
        raise RuntimeError("unexpected block inventory")
    if next(model.parameters()).device.type != "cpu" or torch.get_num_threads() != NATIVE_THREADS:
        raise RuntimeError("device or thread mismatch")
    if {parameter.dtype for parameter in model.parameters()} != {torch.float32}:
        raise RuntimeError("model parameter dtype mismatch")
    if config.nvars != 5 or config.d_model != 128 or config.d_mlp != 512 or config.d_vocab != 16 or config.use_ln:
        raise RuntimeError("runtime configuration mismatch")
    if hasattr(model, "ln_final") or hasattr(model.unembed, "b_U"):
        raise RuntimeError("unexpected final normalization or unembedding bias")
    final = model.blocks[-1]
    hook_points = validate_hook_dependencies(model, final, model_receipt)
    w_out = final.mlp.W_out.detach().cpu().numpy().copy()
    b_out = final.mlp.b_out.detach().cpu().numpy().copy()
    w_u = model.unembed.W_U.detach().cpu().numpy().copy()
    if w_out.shape != (128, 512) or b_out.shape != (128,) or w_u.shape != (128, 16):
        raise RuntimeError("pinned weight convention mismatch")
    if model_receipt.get("checkpoint_sha256") != SOURCE_HASHES[CHECKPOINT]:
        raise RuntimeError("runtime checkpoint receipt mismatch")
    coefficient = w_out.astype(np.float64).T @ w_u.astype(np.float64)
    mlp_bias_logits = b_out.astype(np.float64) @ w_u.astype(np.float64)
    coefficient_check = np.einsum("dj,dv->jv", w_out.astype(np.float64), w_u.astype(np.float64))
    if not np.allclose(coefficient, coefficient_check, atol=1e-12, rtol=1e-12):
        raise RuntimeError("float64 coefficient pre-forward check failed")
    weight_hashes = {name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                     for name, value in {"W_out": w_out, "b_out": b_out, "W_U": w_u}.items()}
    host_fingerprint = native_host_fingerprint(torch)
    save_json_exclusive(forward_marker,
                        {"stage": args.stage,
                         "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         "freeze_sha256": frozen["freeze_sha256"],
                         "coefficient_sha256": hashlib.sha256(coefficient.tobytes(order="C")).hexdigest(),
                         "weight_sha256": weight_hashes,
                         "host_fingerprint": host_fingerprint})
    arrays, input_forwards, batch_calls = extract(model, tokens, ids, replay, hook_points)
    arrays.update(unique_ids=ids, unit_logit_coefficients=coefficient,
                  mlp_bias_logits=mlp_bias_logits, W_out=w_out, b_out=b_out, W_U=w_u)
    reconstructed_resid = arrays["resid_mid"].astype(np.float64) + arrays["post"].astype(np.float64) @ w_out.T.astype(np.float64) + b_out.astype(np.float64)
    reconstructed_logits = arrays["resid_post"].astype(np.float64) @ w_u.astype(np.float64)
    resid_error = float(np.max(np.abs(reconstructed_resid - arrays["resid_post"])))
    logit_error = float(np.max(np.abs(reconstructed_logits - arrays["logits"])))
    if not np.allclose(reconstructed_resid, arrays["resid_post"], atol=ATOL, rtol=RTOL):
        raise RuntimeError("residual reconstruction failed")
    if not np.allclose(reconstructed_logits, arrays["logits"], atol=ATOL, rtol=RTOL):
        raise RuntimeError("logit reconstruction failed")
    if not replay:
        deltas = compare_prior_cache(arrays, prior_arrays)
    else:
        initial_path = HERE / "full_native_outputs.npz"
        initial_receipt = json.loads((HERE / "native_receipt.json").read_text())
        if initial_receipt.get("status") != "PASS" or initial_receipt.get("freeze_sha256") != frozen["freeze_sha256"] or sha256(initial_path) != initial_receipt.get("output_sha256"):
            raise RuntimeError("initial output receipt mismatch")
        with np.load(initial_path, allow_pickle=False) as initial:
            deltas = {key: float(np.max(np.abs(arrays[key].astype(np.float64) - initial[key][ids].astype(np.float64))))
                      for key in ("pre", "post", "resid_mid", "resid_post", "logits")}
        if any(value != 0.0 for value in deltas.values()):
            raise RuntimeError("batch-matched fresh-load replay mismatch")
    save_npz_exclusive(output_path, **arrays)
    receipt = {"status": "PASS", "stage": args.stage, "input_forwards": input_forwards,
               "selected_output_rows": int(len(ids)), "batch_calls": batch_calls,
               "batch_size": NATIVE_BATCH_SIZE, "environment": versions, "model": model_receipt,
               "bank_sha256": sha256(BANK), "output_sha256": sha256(output_path),
               "prior_native_sha256": sha256(PRIOR_NATIVE),
               "preflight": preflight,
               "max_residual_reconstruction_error": resid_error,
               "max_logit_reconstruction_error": logit_error,
               "prior_or_replay_comparison": deltas,
               "weight_sha256": weight_hashes, "host_fingerprint": host_fingerprint,
               "scope": "unchanged checkpoint; complete final MLP state and exact linear readout", **frozen}
    if replay:
        receipt["initial_output_sha256"] = sha256(initial_path)
    save_json_exclusive(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initial", "replay"))
    args = parser.parse_args()
    frozen = verify_frozen_packet()
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__.split("+")[0], "einops": importlib.metadata.version("einops")}
    if versions != NATIVE_ENV:
        raise RuntimeError(f"native environment drift: {versions}")
    host_fingerprint = native_host_fingerprint(torch)
    if host_fingerprint != EXPECTED_NATIVE_HOST:
        raise RuntimeError(f"native host drift: {host_fingerprint}")
    if args.stage == "replay":
        validate_replay_prerequisites(HERE, frozen["freeze_sha256"])
    attempt_path = HERE / f"attempt_{args.stage}.json"
    save_json_exclusive(attempt_path, {"stage": args.stage, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                      "freeze_sha256": frozen["freeze_sha256"], "environment": versions,
                                      "host_fingerprint": host_fingerprint})
    try:
        execute(args, frozen, versions)
    except Exception as error:
        forward_marker = HERE / f"forward_started_{args.stage}.json"
        save_json_exclusive(HERE / f"failure_{args.stage}.json",
                            {"stage": args.stage, "error_type": type(error).__name__, "error": str(error),
                             "traceback": traceback.format_exc(), "freeze_sha256": frozen["freeze_sha256"],
                             "forward_started": forward_marker.exists(),
                             "forward_marker_sha256": sha256(forward_marker) if forward_marker.exists() else None})
        raise


if __name__ == "__main__":
    main()
