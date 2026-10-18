"""Re-execute every protected-bank row from the packaged checkpoint on CPU."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    study = root / "work/next_phase/sat_full_layer_basis_v7_20261018"
    sys.path.insert(0, str(study))
    import protocol
    import native
    protocol.verify_frozen_packet()
    runtime = native.load_runtime()
    model, config, model_receipt = runtime.load_model(threads=2)
    final = model.blocks[-1]
    hooks = {"pre": final.mlp.hook_pre, "post": final.mlp.hook_post,
             "resid_mid": final.hook_resid_mid, "resid_post": final.hook_resid_post}
    with np.load(protocol.BANK, allow_pickle=False) as bank:
        tokens = bank["unique_tokens"].copy()
    accumulated = {key: [] for key in (*hooks, "logits")}
    for start in range(0, len(tokens), protocol.NATIVE_BATCH_SIZE):
        block = native.infer(model, tokens[start:start + protocol.NATIVE_BATCH_SIZE], hooks)
        for key, values in block.items():
            accumulated[key].append(values)
    fresh = {key: np.concatenate(values) for key, values in accumulated.items()}
    comparisons = {}
    with np.load(study / "full_native_outputs.npz", allow_pickle=False) as expected:
        for key, actual in fresh.items():
            reference = expected[key]
            equal = np.array_equal(actual, reference)
            close = np.allclose(actual, reference, atol=1e-6, rtol=1e-6)
            comparisons[key] = {"shape": list(actual.shape), "bitwise_equal": bool(equal),
                                "max_absolute_difference": float(np.max(np.abs(actual.astype(float) - reference.astype(float)))),
                                "within_atol_1e_6_rtol_1e_6": bool(close)}
            if not close:
                raise RuntimeError("fresh native output mismatch: " + key)
        same_bits = np.array_equal(fresh["post"] >= 0.5, expected["post"] >= 0.5)
        same_argmax = np.array_equal(fresh["logits"].argmax(1), expected["logits"].argmax(1))
        if not same_bits or not same_argmax:
            raise RuntimeError("fresh inference changed a threshold event or task prediction")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "fresh_native_outputs.npz").open("xb") as handle:
        np.savez_compressed(handle, **fresh)
    record = {"status": "PASS", "scope": "fresh execution of all 7,252 retained token rows from the packaged checkpoint; no training or new inputs",
              "rows": len(tokens), "batches": (len(tokens) + 255) // 256, "threads": 2,
              "source_checkpoint_sha256": digest(protocol.CHECKPOINT),
              "bank_sha256": digest(protocol.BANK), "script_sha256": digest(__file__),
              "reference_outputs_sha256": digest(study / "full_native_outputs.npz"),
              "fresh_outputs_sha256": digest(args.output / "fresh_native_outputs.npz"),
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "model": model_receipt, "comparisons": comparisons,
              "same_all_512_threshold_bits": bool(same_bits), "same_task_argmax": bool(same_argmax)}
    (args.output / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
