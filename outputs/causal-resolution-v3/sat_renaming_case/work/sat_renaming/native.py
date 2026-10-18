"""Frozen four-unit native readout; full post/residual retained only for affine parity."""
import argparse
import torch
from protocol import *
from runtime import load_model


def infer_block(model, tokens):
    if tokens.shape != (8, 41): raise ValueError("Native batch must be one ordered eight-input matched block")
    cache = {}
    def capture(key, selected=False):
        def hook(tensor, name):
            value = tensor[:, -1, UNITS] if selected else tensor[:, -1, :]
            cache[key] = value.detach().cpu().numpy().copy()
        return hook
    model.blocks[1].mlp.hook_pre.add_hook(capture("pre", True))
    model.blocks[1].mlp.hook_post.add_hook(capture("full_post"))
    model.blocks[1].hook_resid_mid.add_hook(capture("resid_mid"))
    try:
        with torch.inference_mode():
            cache["logits"] = model(torch.from_numpy(np.asarray(tokens, dtype=np.int64)))[:, -1, :].detach().cpu().numpy().copy()
    finally:
        model.remove_all_hooks()
    cache["post"] = cache["full_post"][:, UNITS].copy()
    if not all(np.isfinite(v).all() for v in cache.values()): raise RuntimeError("Nonfinite native output")
    if not np.array_equal(cache["post"], np.maximum(cache["pre"], 0)): raise RuntimeError("Exact ReLU identity failed")
    expected = {"pre": (8, 4), "post": (8, 4), "full_post": (8, 512), "resid_mid": (8, 128), "logits": (8, 16)}
    if any(v.shape != expected[k] or v.dtype != np.float32 for k, v in cache.items()): raise RuntimeError("Unexpected native shape/dtype")
    return cache


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["initial", "replay"])
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= NATIVE_SECONDS: raise ValueError("Invalid remaining phase budget")
    if torch.__version__.split("+")[0] != "2.1.2" or np.__version__ != "1.26.4": raise RuntimeError("Pinned native runtime drift")
    require_environment("native")
    selected_frozen()
    start, cpu = time.monotonic(), time.process_time()
    save_json("native_" + args.stage + "_started.json", {"utc": utc(), "selected_freeze_sha256": sha(ROOT / "selected_data_freeze.json"), "remaining_seconds": args.seconds})
    if args.stage == "replay":
        receipt = read_json("native_initial.json")
        if receipt["status"] != "PASS": raise RuntimeError("No successful initial extraction")
        require_hash(ROOT / "native_outputs.npz", receipt["output_sha256"])
    with np.load(ROOT / "selected_bank.npz", allow_pickle=False) as bank: tokens = bank["tokens"].copy()
    model, config, source = load_model(threads=2)
    weights = {"W_out": model.blocks[1].mlp.W_out.detach().cpu().numpy().copy(),
               "b_out": model.blocks[1].mlp.b_out.detach().cpu().numpy().copy(),
               "W_U": model.unembed.W_U.detach().cpu().numpy().copy()}
    if args.stage == "initial": save_npz("readout_weights.npz", **weights)
    else:
        require_hash(ROOT / "readout_weights.npz", receipt["weights_sha256"])
        with np.load(ROOT / "readout_weights.npz", allow_pickle=False) as saved:
            if any(not np.array_equal(weights[k], saved[k]) for k in weights): raise RuntimeError("Readout changed on replay")
    block_ids = list(range(64)) if args.stage == "initial" else read_json("selected_bank.json")["replay_block_ids"]
    directory = ROOT / ("native_batches" if args.stage == "initial" else "replay_batches")
    directory.mkdir(exist_ok=False)
    journal = "native_" + args.stage + "_events.jsonl"
    with (ROOT / journal).open("x"): pass
    gathered = {k: [] for k in ["pre", "post", "full_post", "resid_mid", "logits"]}
    replay_checks = []
    saved = np.load(ROOT / "native_outputs.npz", allow_pickle=False) if args.stage == "replay" else None
    try:
        for block in block_ids:
            deadline(start, args.seconds, "native " + args.stage)
            phase = time.monotonic()
            append_event(journal, {"event": "BATCH_ATTEMPT", "utc": utc(), "block_id": block,
                                  "conservative_input_forward_charge": 8, "batch_call_charge": 1})
            value = infer_block(model, tokens[block].reshape(8, 41))
            path = directory / ("block_%03d.npz" % block)
            with path.open("xb") as f: np.savez_compressed(f, **value)
            for k, v in value.items(): gathered[k].append(v.reshape(2, 4, -1))
            event = {"event": "BATCH_COMPLETE", "block_id": block, "input_forwards": 8, "batch_calls": 1, "elapsed_seconds": time.monotonic() - phase, "sha256": sha(path)}
            if saved is not None:
                differences = {k: float(np.max(np.abs(value[k].astype(np.float64) - saved[k][block].reshape(8, -1)))) for k in value}
                compatible = all(np.allclose(value[k], saved[k][block].reshape(8, -1), atol=ATOL, rtol=RTOL) for k in value)
                same_bits = np.array_equal(value["post"] >= .5, saved["post"][block].reshape(8, 4) >= .5)
                same_labels = np.array_equal(value["logits"].argmax(axis=-1), saved["logits"][block].reshape(8, 16).argmax(axis=-1))
                replay_checks.append({"block_id": block, "compatible": bool(compatible), "same_bits": bool(same_bits), "same_full_argmax": bool(same_labels), "max_absolute_differences": differences})
                event["replay"] = replay_checks[-1]
            append_event(journal, event)
        deadline(start, args.seconds, "native " + args.stage)
        arrays = {k: np.stack(v) for k, v in gathered.items()}
        arrays["block_ids"] = np.asarray(block_ids, dtype=np.int16)
        arrays["units"] = UNITS.copy()
        arrays["post_distance_to_half"] = np.abs(arrays["post"].astype(np.float64) - .5)
        ordered = np.sort(arrays["logits"].astype(np.float64), axis=-1)
        arrays["winning_logit_gap"] = ordered[..., -1] - ordered[..., -2]
        filename = "native_outputs.npz" if args.stage == "initial" else "replay_outputs.npz"
        save_npz(filename, **arrays)
        passed = all(r["compatible"] and r["same_bits"] and r["same_full_argmax"] for r in replay_checks)
        save_json("native_" + args.stage + ".json", {"status": "PASS" if passed else "REPLAY_FAIL", "utc": utc(), "source": source,
                  "input_forwards": 8 * len(block_ids), "batch_calls": len(block_ids), "batch_size": 8,
                  "block_ids": block_ids, "cost": cost(start, cpu), "output_sha256": sha(ROOT / filename),
                  "weights_sha256": sha(ROOT / "readout_weights.npz"), "checks": replay_checks,
                  "scope": "Unmodified published native CPU float32; fixed units only. Full post/residual used solely for affine parity."})
    finally:
        if saved is not None: saved.close()
    print(json.dumps(read_json("native_" + args.stage + ".json")))


if __name__ == "__main__": main()
