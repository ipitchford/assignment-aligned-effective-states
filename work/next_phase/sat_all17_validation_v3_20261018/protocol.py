"""Constants and pure semantic helpers for the all-17 SAT validation."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]

UNITS = np.array([10, 41, 55, 77, 81, 185, 224, 245, 261, 291, 317, 326,
                  380, 411, 416, 492, 495], dtype=np.int16)
ASSIGNMENT_IDS = np.array([16, 0, 28, 7, 14, 3, 1, 9, 27, 23, 17, 30,
                           19, 25, 28, 21, 2], dtype=np.int8)
ASSIGNMENT_NAMES = ("TFFFF", "FFFFF", "TTTFF", "FFTTT", "FTTTF", "FFFTT",
                    "FFFFT", "FTFFT", "TTFTT", "TFTTT", "TFFFT", "TTTTF",
                    "TFFTT", "TTFFT", "TTTFF", "TFTFT", "FFFTF")
UNIQUE_ASSIGNMENTS = np.array(sorted(set(map(int, ASSIGNMENT_IDS))), dtype=np.int8)
STRATA = ("sole_positive", "multi_positive", "unsat_negative",
          "sole_negative", "multi_negative")
POSITIVE_STRATA = (0, 1)
NEGATIVE_STRATA = (2, 3, 4)
TRANSPOSITIONS = tuple(itertools.combinations(range(5), 2))
ALL_VARIABLE_PERMUTATIONS = tuple(itertools.permutations(range(5)))

SAMPLE_SEED = 2026101807
BOOTSTRAP_SEED = 2026101808
PROPOSALS = 250_000
QUOTA = 32
TRANSFORM_QUOTA = 8
BOOTSTRAP_DRAWS = 2_000
SOURCE_THRESHOLD = 0.5
ATOL = 1e-4
RTOL = 1e-5
ANALYSIS_PYTHON = "3.14.7"
ANALYSIS_NUMPY = "2.4.2"
NATIVE_PYTHON = "3.11.14"
NATIVE_NUMPY = "1.26.4"
NATIVE_TORCH = "2.1.2"
NATIVE_EINOPS = "0.8.1"
RUNTIME_SHA256 = "bbc76d5f2d4a856944f54269a6da2d388f0ea6aee81ab03d9486e2e5c8bca230"
PRIOR_LEDGER_SHA256 = "8fb9d138443bb4d72a0265196343199ccb665c08b34a81737d75e0eb02ee29eb"
PRIOR_RENAMING_SHA256 = "4c4e668f80328ed34349448c1506c0702bce0a35464c0896a9b9181d9d2adf8a"
PRIOR_LEDGER_KEYS = ("tokens",)
PRIOR_RENAMING_KEYS = ("clauses", "tokens", "counts", "labels", "expected_unit_bits", "pair_index", "trial_ids")
PRIOR_LEDGER_TOKEN_ROWS = 8957
PRIOR_RENAMING_TOKEN_ROWS = 512
SAT_TOKEN = 13
UNSAT_TOKEN = 14
HELPERS_SHA256 = "ece43db7317fa8e3ad6cb6fb40bb1ba9f98ecd9191cbec6689b0bc67c0a268fe"
INTERPRETATION_SHA256 = "b8f87ceaecb0c2449f3f626983ec028f2d3661c394d08f7d96641f4e06ba2177"
SOURCE_MAPPING_SHA256 = "5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d"
FREEZE_SCHEMA = "sat-all17-static-freeze-v3"
FREEZE_ENV = "SAT_ALL17_FREEZE_SHA256"
FROZEN_FILES = (
    "protocol.md", "protocol.py", "construct.py", "verify_bank.py", "native.py",
    "analyze.py", "selfcheck.py", "selfcheck.json", "pilot_common.py",
    "runtime_source_snapshot.py", "source_bindings.md",
)

MODEL_DIR = WORKSPACE / "work" / "sat_case"
CHECKPOINT = MODEL_DIR / "upstream/models/layers_2_heads_1_4_1M/900.pth"
PRIOR_LEDGER = WORKSPACE / "outputs/causal-resolution-v3/sat_renaming_case/work/sat_renaming_launch_recovery/prior_ledger_tokens.npz"
PRIOR_RENAMING = WORKSPACE / "outputs/causal-resolution-v3/sat_renaming_case/work/sat_renaming_launch_recovery/selected_bank.npz"
HELPERS_SOURCE = MODEL_DIR / "upstream/helpers.py"
INTERPRETATION_SOURCE = MODEL_DIR / "upstream/interpretation.py"
SOURCE_MAPPING = WORKSPACE / "outputs/causal-resolution-v3/sat_mechanism/source_mapping.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def domain_hash(domain: str, *parts: bytes) -> bytes:
    h = hashlib.sha256(domain.encode("ascii") + b"\0")
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.digest()


def verify_frozen_packet() -> dict:
    """Bind an execution stage to the externally reviewed freeze manifest."""
    expected = os.environ.get(FREEZE_ENV, "")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise RuntimeError(f"{FREEZE_ENV} must contain the reviewed freeze SHA-256")
    freeze_path = HERE / "freeze.json"
    actual = sha256(freeze_path)
    if actual != expected:
        raise RuntimeError("externally anchored freeze hash mismatch")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise RuntimeError("freeze schema mismatch")
    rows = freeze.get("files", [])
    if tuple(row.get("path") for row in rows) != FROZEN_FILES:
        raise RuntimeError("frozen file inventory/order mismatch")
    code_sha256 = {}
    for row in rows:
        path = HERE / row["path"]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("frozen file mismatch: " + row["path"])
        code_sha256[row["path"]] = row["sha256"]
    selfcheck = json.loads((HERE / "selfcheck.json").read_text())
    if selfcheck.get("status") != "PASS":
        raise RuntimeError("frozen self-check is not passing")
    if sha256(HELPERS_SOURCE) != HELPERS_SHA256 or sha256(INTERPRETATION_SOURCE) != INTERPRETATION_SHA256:
        raise RuntimeError("published vocabulary or assignment source hash mismatch")
    if sha256(SOURCE_MAPPING) != SOURCE_MAPPING_SHA256:
        raise RuntimeError("published unit/assignment mapping hash mismatch")
    mapping = json.loads(SOURCE_MAPPING.read_text())["single_assignment_general_units"]
    observed = [(int(row["unit"]), row["assignment"], int(row["truth_table_index"])) for row in mapping]
    expected_mapping = list(zip(map(int, UNITS), ASSIGNMENT_NAMES, map(int, ASSIGNMENT_IDS)))
    if observed != expected_mapping:
        raise RuntimeError("published unit/assignment mapping content mismatch")
    return {"freeze_sha256": actual, "code_sha256": code_sha256,
            "selfcheck_sha256": code_sha256["selfcheck.json"],
            "source_sha256": {"helpers.py": HELPERS_SHA256,
                              "interpretation.py": INTERPRETATION_SHA256,
                              "source_mapping.json": SOURCE_MAPPING_SHA256}}


def classify_stratum(counts: np.ndarray, assignment: int) -> int | None:
    nsol = int(np.count_nonzero(counts == 0))
    target = int(counts[assignment]) == 0
    if nsol == 1 and target:
        return 0
    if nsol >= 2 and target:
        return 1
    if nsol == 0:
        return 2
    if nsol == 1 and not target:
        return 3
    if nsol >= 2 and not target:
        return 4
    return None


def min_solution_hamming(counts: np.ndarray, assignment: int) -> int | None:
    solutions = np.flatnonzero(np.asarray(counts) == 0)
    if not len(solutions):
        return None
    return min((int(solution) ^ int(assignment)).bit_count() for solution in solutions)


def swap_assignment(assignment: int, pair: tuple[int, int]) -> int:
    bits = [(assignment >> (4 - i)) & 1 for i in range(5)]
    i, j = pair
    bits[i], bits[j] = bits[j], bits[i]
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def exact_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires both classes")
    delta = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(delta > 0) + 0.5 * np.count_nonzero(delta == 0)) / delta.size)
