"""Fixed SAT-renaming protocol. Importing this module performs no study sampling."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PILOT = WORK / "sat_case"
sys.path.insert(0, str(PILOT))
from pilot_common import CLAUSES, ASSIGNMENTS, LITERAL_TRUTH, CLAUSE_TRUTH, encode, semantic_counts

CONTRACT = WORK / "next_phase/sat_renaming_execution_contract.md"
CONTRACT_SHA = "4b9dc2df3002dbbe56cbcc5d38c119ace801f16e2f3bd018d345deb4eda7af8c"
ENCODER_SHA = "5f0f54e797ffadeb7b802fec07eae425a877ce7cfed4a5c9e9353ca91a939c61"
CHECKPOINT = PILOT / "upstream/models/layers_2_heads_1_4_1M/900.pth"
CHECKPOINT_SHA = "d587446c64b7a8d68de93808ccf84a84364b85dcaf2149b8983b77895e24046d"
MODEL_SHA = "7555e1f2f27546225397aa12de4fde1f6c903e9804766de477fe4c02770691ea"
RUNTIME_SHA = "bbc76d5f2d4a856944f54269a6da2d388f0ea6aee81ab03d9486e2e5c8bca230"
MAPPING_SHA = "5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d"
UNITS = np.array([261, 291, 411, 492], dtype=np.int64)
ASSIGNMENT_IDS = np.array([27, 23, 25, 21], dtype=np.int64)
PAIRS = [("P1", 27, 23), ("P2", 25, 21)]
PERMUTATIONS = {"correct": np.array([1, 0, 3, 2]), "identity": np.arange(4), "wrong": np.array([3, 2, 1, 0])}
COHORT_ORDER = ["positive", "negative"]
STATE_ORDER = ["original", "edited", "renamed_original", "renamed_edited"]
ACTION_FIELDS = ["slot", "old_clause_id", "kept_literal", "new_variable", "new_clause_id"]
CLAUSE_LOOKUP = {tuple(map(int, row)): i for i, row in enumerate(CLAUSES)}
ALLOWED = np.array([i for i, c in enumerate(CLAUSES) if set(int(t) % 5 for t in c) != {1, 2}], dtype=np.int16)
LITERAL_RENAME = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9], dtype=np.int16)
CLAUSE_RENAME = np.array([CLAUSE_LOOKUP[tuple(sorted((int(LITERAL_RENAME[t]) for t in c), key=lambda t: t % 5))] for c in CLAUSES], dtype=np.int16)
SAMPLE_SEED, BOOTSTRAP_SEED = 2026101604, 2026101605
PROPOSALS, QUOTA, BOOTSTRAP_DRAWS = 100_000, 32, 2_000
CONSTRUCTION_SECONDS, NATIVE_SECONDS, ANALYSIS_SECONDS = 1200, 600, 300
ATOL, RTOL = 1e-4, 1e-5
ANALYSIS_NUMPY = "2.4.6"
LEDGER_SOURCES = [
    ("sat_case/pilot_bank.npz", "tokens", "tokens"),
    ("sat_mechanism/query_tokens.npz", "unique_tokens", "tokens"),
    ("sat_case/candidate_logits.npz", "edited_clauses", "clause_ids"),
    ("sat_case/selected_decision_check.npz", "edited_clauses", "clause_ids"),
    ("sat_case/primary_witness.npz", "original_tokens", "tokens"),
    ("sat_case/primary_witness.npz", "edited_tokens", "tokens"),
]
IMPLEMENTATION_FILES = ["protocol.py", "prepare_freeze.py", "construct.py", "native.py", "analyze.py", "run_study.py", "independent_oracle.py", "verify_bank.py", "selfcheck.py", "selfcheck.json", "requirements-analysis.txt", "requirements-native.txt", "environment.json", "README.md"]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def save_json(name, value):
    path = ROOT / name
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=jsonable, allow_nan=False)
        f.write("\n")


def read_json(name):
    return json.loads((ROOT / name).read_text())


def save_npz(name, **arrays):
    with (ROOT / name).open("xb") as f:
        np.savez_compressed(f, **arrays)


def append_event(path, event):
    with (ROOT / path).open("a") as f:
        f.write(json.dumps(event, sort_keys=True, default=jsonable, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def record(path):
    path = Path(path)
    return {"path": os.path.relpath(path, ROOT), "sha256": sha(path), "bytes": path.stat().st_size}


def require_hash(path, expected):
    if sha(path) != expected: raise RuntimeError("Frozen SHA256 mismatch: " + str(path))


def frozen():
    require_hash(CONTRACT, CONTRACT_SHA)
    data = read_json("freeze.json")
    if data["contract_sha256"] != CONTRACT_SHA: raise RuntimeError("Wrong contract in freeze")
    for row in data["files"]: require_hash(ROOT / row["path"], row["sha256"])
    return data


def selected_frozen():
    frozen()
    data = read_json("selected_data_freeze.json")
    if data["implementation_freeze_sha256"] != sha(ROOT / "freeze.json"): raise RuntimeError("Selection/freezing chronology mismatch")
    for row in data["files"]: require_hash(ROOT / row["path"], row["sha256"])
    if read_json("construction.json")["status"] != "PASS": raise RuntimeError("Construction gate not passed")
    if read_json("bank_verification.json").get("status") != "PASS": raise RuntimeError("Independent semantic bank verification not passed")
    return data


def require_analysis_numpy():
    if np.__version__ != ANALYSIS_NUMPY: raise RuntimeError("Analysis NumPy version drift: " + np.__version__)
    if (ROOT / "environment.json").exists(): require_environment("analysis")


def require_environment(kind):
    declared = read_json("environment.json")[kind]
    if sys.version.split()[0] != declared["python"]: raise RuntimeError("Python version drift")
    expected_executable = declared.get("executable") or str(ROOT / declared["executable_relative"])
    if Path(sys.executable).absolute().resolve() != Path(expected_executable).absolute().resolve(): raise RuntimeError("Interpreter identity drift")
    for name in ["numpy", "torch", "einops"]:
        if name in declared and importlib.metadata.version(name).split("+")[0] != declared[name]: raise RuntimeError("Package version drift: " + name)


def token_bytes(row):
    row = np.asarray(row)
    if row.shape != (41,) or not np.issubdtype(row.dtype, np.integer) or np.any((row < 0) | (row > 15)):
        raise ValueError("Invalid token identity row")
    return bytes(int(x) for x in row)


def decode_tokens(rows):
    rows = np.asarray(rows)
    if rows.ndim == 1: rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 41: raise ValueError("Bad token shape")
    if not (np.all(rows[:, 0:40:4] == 10) and np.all(rows[:, 3:40:4] == 11) and np.all(rows[:, 40] == 12)):
        raise ValueError("Invalid native formula syntax")
    clauses = np.array([[CLAUSE_LOOKUP[(int(row[1 + 4*k]), int(row[2 + 4*k]))] for k in range(10)] for row in rows], dtype=np.int16)
    if not np.array_equal(encode(clauses), rows): raise ValueError("Canonical encoder roundtrip failed")
    return clauses


def rename_clauses(clauses):
    return CLAUSE_RENAME[np.asarray(clauses)]


def orbit_bytes(clauses):
    original = token_bytes(encode(np.asarray(clauses)[None, :])[0])
    renamed = token_bytes(encode(rename_clauses(clauses)[None, :])[0])
    return min(original, renamed)


def digest(domain, *parts):
    """UTF8 domain + NUL, then 8-byte unsigned big-endian length and raw bytes per part."""
    h = hashlib.sha256(domain.encode("ascii") + b"\0")
    for part in parts:
        h.update(len(part).to_bytes(8, "big")); h.update(part)
    return h.digest()


def action_dict(action):
    return dict(zip(ACTION_FIELDS, map(int, action)))


def quartet(clauses, action):
    changed = np.array(clauses, dtype=np.int16, copy=True)
    changed[int(action[0])] = int(action[4])
    return np.stack([clauses, changed, rename_clauses(clauses), rename_clauses(changed)])


def cost(start_wall, start_cpu):
    # macOS ru_maxrss is bytes; Linux reports KiB.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"elapsed_seconds": time.monotonic() - start_wall, "cpu_seconds": time.process_time() - start_cpu,
            "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
            "python": sys.version, "numpy": np.__version__}


def deadline(start, seconds, phase):
    if time.monotonic() - start > seconds: raise TimeoutError(phase + " fixed wall cap reached")
