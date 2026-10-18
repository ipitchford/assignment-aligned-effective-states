"""Build the historical-input ledger and bind the COMPLETE implementation before sampling."""
import argparse
import platform
from protocol import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-freeze", action="store_true", required=True)
    parser.parse_args()
    if any((ROOT / p).exists() for p in ["freeze.json", "freeze_started.json", "construction_started.json"]):
        raise RuntimeError("Refusing repeat freeze or a late freeze")
    require_analysis_numpy()
    for name in IMPLEMENTATION_FILES:
        if not (ROOT / name).is_file(): raise RuntimeError("Missing complete implementation: " + name)
    selfcheck = read_json("selfcheck.json")
    if selfcheck.get("status") != "PASS": raise RuntimeError("Deterministic self-check did not pass")
    for name, expected in selfcheck["tested_hashes"].items(): require_hash(ROOT/name, expected)
    for path, expected in [(CONTRACT, CONTRACT_SHA), (PILOT / "pilot_common.py", ENCODER_SHA),
                           (PILOT / "runtime.py", RUNTIME_SHA), (PILOT / "upstream/model.py", MODEL_SHA),
                           (CHECKPOINT, CHECKPOINT_SHA), (WORK / "sat_mechanism/source_mapping.json", MAPPING_SHA)]:
        require_hash(path, expected)
    save_json("freeze_started.json", {"utc": utc(), "scope": "Historical ledger and implementation freeze only; no proposals, native forwards or scientific draws"})
    sources, rows = [], set()
    from independent_oracle import decode_tokens as independent_decode
    for relative, key, kind in LEDGER_SOURCES:
        path = WORK / relative
        with np.load(path, allow_pickle=False) as data: original = np.array(data[key], copy=True)
        if kind == "clause_ids":
            clauses = original.reshape(-1, 10)
            tokens = encode(clauses)
        else:
            tokens = original.reshape(-1, 41)
            clauses = decode_tokens(tokens)
        if not np.array_equal(encode(clauses), tokens): raise RuntimeError("Prior encoder disagreement")
        for row in tokens:
            independent_decode([int(t) for t in row])
            rows.add(token_bytes(row))
        sources.append({"path": relative, "key": key, "kind": kind, "sha256": sha(path),
                        "row_count": len(tokens), "original_shape": list(original.shape)})
    tokens = np.array([list(row) for row in sorted(rows)], dtype=np.int64)
    save_npz("prior_ledger_tokens.npz", tokens=tokens)
    save_json("prior_ledger.json", {"schema_version": 1, "sources": sources,
              "encoder": {"path": "sat_case/pilot_common.py", "sha256": ENCODER_SHA},
              "raw_rows": sum(s["row_count"] for s in sources), "unique_rows": len(tokens),
              "identity_encoding": "41 raw uint8 token bytes, lexicographic sorted deduplicated rows",
              "historical_smoke": "One invalid-grammar 41-zero shape-smoke row is excluded; it cannot equal a valid formula.",
              "ledger_sha256": sha(ROOT / "prior_ledger_tokens.npz")})
    paths = [ROOT / name for name in IMPLEMENTATION_FILES]
    paths += [CONTRACT, PILOT / "pilot_common.py", PILOT / "runtime.py", PILOT / "upstream/model.py", CHECKPOINT,
              WORK / "sat_mechanism/source_mapping.json",
              WORK / "next_phase/sat_invariance_admission.md", WORK / "next_phase/sat_renaming_design_draft.md",
              WORK / "next_phase/sat_renaming_adversarial_admission.md", WORK / "next_phase/sat_renaming_final_contract_check.md",
              WORK / "next_phase/sat_analysis_code_review.md", WORK / "next_phase/sat_semantic_theory.md"]
    paths += [WORK / row[0] for row in LEDGER_SOURCES]
    paths += [ROOT / "prior_ledger_tokens.npz", ROOT / "prior_ledger.json"]
    paths = sorted(set(paths), key=lambda p: os.path.relpath(p, ROOT))
    save_json("freeze.json", {"utc": utc(), "contract_sha256": CONTRACT_SHA,
              "files": [record(p) for p in paths], "allowed_clause_ids": ALLOWED.tolist(),
              "sample_seed": SAMPLE_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "numpy": np.__version__,
              "platform": platform.platform(), "python": sys.version,
              "scope": "Complete preconstruction implementation and historical-input ledger. No neural or candidate outputs existed."})
    frozen()
    print(json.dumps({"status": "FROZEN", "sha256": sha(ROOT / "freeze.json"), "prior_unique_rows": len(tokens)}))


if __name__ == "__main__": main()
