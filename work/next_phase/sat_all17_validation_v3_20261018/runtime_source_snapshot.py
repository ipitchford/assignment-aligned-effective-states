"""Lean CPU loader for the unmodified published Palumbo et al. 2-SAT model.

Only the exact original class ASTs are executed. Unrelated plotting, notebook,
training, and dataset imports are not executed. No model computation is changed.
Run this file directly for a shape-only smoke check on a non-formula dummy input.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import types
from dataclasses import dataclass, field
from typing import Optional, List

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
MODEL_SOURCE = UPSTREAM / "model.py"
CHECKPOINT = UPSTREAM / "models/layers_2_heads_1_4_1M/900.pth"
MODEL_SHA256 = "7555e1f2f27546225397aa12de4fde1f6c903e9804766de477fe4c02770691ea"
CHECKPOINT_SHA256 = "d587446c64b7a8d68de93808ccf84a84364b85dcaf2149b8983b77895e24046d"
CLASS_NAMES = (
    "Config", "HookPoint", "Embed", "Unembed", "PosEmbed", "LayerNorm",
    "Attention", "MLP", "TransformerBlock", "TransformerBlockNoMLP", "Transformer",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_classes():
    if sha256(MODEL_SOURCE) != MODEL_SHA256:
        raise ValueError("Published model.py SHA256 mismatch")
    tree = ast.parse(MODEL_SOURCE.read_text(), filename=str(MODEL_SOURCE))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in CLASS_NAMES]
    if [n.name for n in classes] != list(CLASS_NAMES):
        raise ValueError("Original class inventory/order changed")
    module = types.ModuleType("sat_published_original_classes")
    module.__dict__.update(
        np=np, t=torch, nn=nn, F=F, einops=einops, math=math, os=os,
        dataclass=dataclass, field=field, Optional=Optional, List=List,
    )
    sys.modules[module.__name__] = module
    exact_tree = ast.Module(body=classes, type_ignores=[])
    exec(compile(exact_tree, str(MODEL_SOURCE), "exec", dont_inherit=True), module.__dict__)
    class_hashes = {
        n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
        for n in classes
    }
    return module, class_hashes


def load_model(threads=2):
    """Load all published state entries strictly on CPU, with no dataset access."""
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Published checkpoint SHA256 mismatch")
    torch.set_num_threads(threads)
    torch.manual_seed(0)  # Initial values are entirely replaced by strict load.
    original, class_hashes = original_classes()
    config = original.Config(
        device=torch.device("cpu"), num_heads=[1, 4],
        load_model_path=str(CHECKPOINT),
    )
    model = original.Transformer(config, use_cache=False, use_ln=False)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Unexpected checkpoint structure")
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Checkpoint mismatch despite strict load")
    model.eval()
    model.requires_grad_(False)
    metadata = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_source_sha256": MODEL_SHA256,
        "original_class_ast_sha256": class_hashes,
        "checkpoint_keys": sorted(payload),
        "checkpoint_epoch": payload.get("epoch"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "state_entries": len(payload["model"]),
        "strict_state_load": True,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "dtype": str(next(model.parameters()).dtype),
    }
    return model, config, metadata


def smoke():
    model, config, receipt = load_model()
    cache = {}
    model.cache_all(cache)
    # Intentionally invalid grammar: this is not a selected scientific test case.
    dummy = torch.zeros((1, config.n_ctx), dtype=torch.long)
    with torch.inference_mode():
        output = model(dummy)
    model.remove_all_hooks()
    assert tuple(output.shape) == (1, 41, 16)
    assert bool(torch.isfinite(output).all())
    expected = {
        "blocks.0.hook_resid_post": (1, 41, 128),
        "blocks.1.hook_resid_mid": (1, 41, 128),
        "blocks.1.mlp.hook_post": (1, 41, 512),
    }
    for key, shape in expected.items():
        assert tuple(cache[key].shape) == shape
    receipt.update(
        status="PASS", scope="strict load and tensor-shape-only smoke; no valid formula or causal experiment",
        dummy_input="one 41-token all-zero row, outside formula grammar",
        output_shape=list(output.shape), all_output_entries_finite=True,
        inspected_hook_shapes={k: list(cache[k].shape) for k in expected},
        python=platform.python_version(), machine=platform.machine(),
        dependencies={k: importlib.metadata.version(k) for k in ("torch", "numpy", "einops")},
        adapter_sha256=sha256(__file__),
    )
    target = ROOT / "runtime_smoke.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "receipt": str(target), "parameter_count": receipt["parameter_count"]}))


if __name__ == "__main__":
    smoke()
