"""One bounded native extraction from the unchanged published checkpoint."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import numpy as np
import torch

from protocol import *

if sha256(MODEL_DIR / "runtime.py") != RUNTIME_SHA256:
    raise RuntimeError("pinned runtime.py SHA-256 mismatch")
runtime_spec = importlib.util.spec_from_file_location("sat_all17_pinned_runtime", MODEL_DIR / "runtime.py")
if runtime_spec is None or runtime_spec.loader is None:
    raise RuntimeError("cannot construct pinned runtime import")
runtime_module = importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(runtime_module)
load_model = runtime_module.load_model


def infer(model, token_rows: np.ndarray) -> dict[str, np.ndarray]:
    cache = {}
    def capture(key, relu=False):
        def hook(tensor, name):
            value = tensor[:, -1, UNITS.astype(np.int64)]
            cache[key] = value.detach().cpu().numpy().copy()
        return hook
    model.blocks[1].mlp.hook_pre.add_hook(capture("pre"))
    model.blocks[1].mlp.hook_post.add_hook(capture("post", True))
    try:
        with torch.inference_mode():
            cache["logits"] = model(torch.from_numpy(np.asarray(token_rows, dtype=np.int64)))[:, -1, :].detach().cpu().numpy().copy()
    finally:
        model.remove_all_hooks()
    if not np.array_equal(cache["post"], np.maximum(cache["pre"], 0)):
        raise RuntimeError("exact ReLU identity failed")
    if not all(value.dtype == np.float32 and np.isfinite(value).all() for value in cache.values()):
        raise RuntimeError("native dtype or finiteness failure")
    return cache


def save_npz_exclusive(path, **arrays):
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)


def save_json_exclusive(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initial", "replay"))
    args = parser.parse_args()
    frozen = verify_frozen_packet()
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__.split("+")[0],
        "einops": importlib.metadata.version("einops"),
    }
    expected_versions = {
        "python": NATIVE_PYTHON, "numpy": NATIVE_NUMPY,
        "torch": NATIVE_TORCH, "einops": NATIVE_EINOPS,
    }
    if versions != expected_versions:
        raise RuntimeError("pinned native environment drift")
    construction = json.loads((HERE / "construction.json").read_text())
    verification = json.loads((HERE / "bank_verification.json").read_text())
    if construction.get("status") != "PASS" or verification.get("status") != "PASS":
        raise RuntimeError("construction or semantic verification not admitted")
    for label, receipt in (("construction", construction), ("verification", verification)):
        if any(receipt.get(key) != frozen[key] for key in ("freeze_sha256", "code_sha256", "selfcheck_sha256", "source_sha256")):
            raise RuntimeError(f"{label} used a different frozen code packet")
    if sha256(HELPERS_SOURCE) != HELPERS_SHA256 or sha256(INTERPRETATION_SOURCE) != INTERPRETATION_SHA256:
        raise RuntimeError("published vocabulary or assignment source hash mismatch")
    selected_bank_hash = sha256(HERE / "selected_bank.npz")
    if construction.get("selected_bank_sha256") != selected_bank_hash:
        raise RuntimeError("construction receipt does not bind selected bank")
    if verification.get("selected_bank_sha256") != selected_bank_hash:
        raise RuntimeError("verification receipt does not bind selected bank")
    with np.load(HERE / "selected_bank.npz", allow_pickle=False) as z:
        tokens = z["unique_tokens"].copy()
    ids = np.arange(len(tokens), dtype=np.int32)
    if args.stage == "replay":
        ids = np.asarray(construction["replay_unique_ids"], dtype=np.int32)
    output_path = HERE / ("native_outputs.npz" if args.stage == "initial" else "replay_outputs.npz")
    receipt_path = HERE / ("native_receipt.json" if args.stage == "initial" else "replay_receipt.json")
    if output_path.exists() or receipt_path.exists():
        raise RuntimeError("exclusive native outputs already exist")
    original_arrays = None
    original_output_hash = None
    if args.stage == "replay":
        native_receipt_path = HERE / "native_receipt.json"
        native_output_path = HERE / "native_outputs.npz"
        if not native_receipt_path.exists() or not native_output_path.exists():
            raise RuntimeError("initial native artifacts required before replay")
        native_receipt = json.loads(native_receipt_path.read_text())
        original_output_hash = sha256(native_output_path)
        if native_receipt.get("status") != "PASS":
            raise RuntimeError("initial native receipt not passed")
        if native_receipt.get("selected_bank_sha256") != selected_bank_hash:
            raise RuntimeError("initial native receipt selected-bank mismatch")
        if any(native_receipt.get(key) != frozen[key] for key in ("freeze_sha256", "code_sha256", "selfcheck_sha256", "source_sha256")):
            raise RuntimeError("initial native receipt used a different frozen code packet")
        if native_receipt.get("output_sha256") != original_output_hash:
            raise RuntimeError("initial native receipt output mismatch")
        with np.load(native_output_path, allow_pickle=False) as original:
            original_arrays = {key: original[key].copy() for key in original.files}
    model, _, model_receipt = load_model(threads=2)
    gathered = {"pre": [], "post": [], "logits": []}
    batch_size = 256
    for start in range(0, len(ids), batch_size):
        block = ids[start:start + batch_size]
        value = infer(model, tokens[block])
        for key in gathered:
            gathered[key].append(value[key])
    arrays = {key: np.concatenate(value, axis=0) for key, value in gathered.items()}
    arrays["unique_ids"] = ids
    arrays["distance_to_half"] = np.abs(arrays["pre"].astype(np.float64) - SOURCE_THRESHOLD)
    ordered = np.sort(arrays["logits"].astype(np.float64), axis=-1)
    arrays["winning_logit_gap"] = ordered[:, -1] - ordered[:, -2]
    save_npz_exclusive(output_path, **arrays)
    receipt = {
        "status": "PASS",
        "stage": args.stage,
        "input_forwards": int(len(ids)),
        "batch_calls": int((len(ids) + batch_size - 1) // batch_size),
        "batch_size_ceiling": batch_size,
        "selected_bank_sha256": selected_bank_hash,
        "output_sha256": sha256(output_path),
        "runtime_sha256": sha256(MODEL_DIR / "runtime.py"),
        "environment": versions,
        "source_bindings": {"helpers_py_sha256": sha256(HELPERS_SOURCE),
                            "interpretation_py_sha256": sha256(INTERPRETATION_SOURCE)},
        "model": model_receipt,
        "scope": "unchanged published CPU float32 checkpoint; seventeen source-nominated units and full task logits",
        **frozen,
    }
    if args.stage == "replay":
        replay_bits = arrays["post"] >= SOURCE_THRESHOLD
        source_bits = original_arrays["post"][ids] >= SOURCE_THRESHOLD
        replay_argmax = arrays["logits"].argmax(axis=-1)
        source_argmax = original_arrays["logits"][ids].argmax(axis=-1)
        pre_delta = np.abs(arrays["pre"].astype(np.float64) - original_arrays["pre"][ids].astype(np.float64))
        receipt["native_outputs_sha256"] = original_output_hash
        receipt["same_threshold_bits"] = bool(np.array_equal(replay_bits, source_bits))
        receipt["same_task_argmax"] = bool(np.array_equal(replay_argmax, source_argmax))
        receipt["preactivation_allclose"] = bool(np.allclose(
            arrays["pre"], original_arrays["pre"][ids], atol=ATOL, rtol=RTOL,
        ))
        receipt["max_absolute_pre_difference"] = float(np.max(pre_delta))
        if not receipt["same_threshold_bits"] or not receipt["same_task_argmax"]:
            receipt["status"] = "REPLAY_FAIL"
    save_json_exclusive(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
