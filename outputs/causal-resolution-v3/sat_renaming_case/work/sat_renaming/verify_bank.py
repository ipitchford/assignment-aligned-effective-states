"""Independent semantic/exclusion verifier, run before any native extraction.

Only saved inputs are read. No sampler, vectorized truth routine, model or
checkpoint loader is imported. NumPy is used solely to read NPZ arrays and check
their storage schema; every semantic value is obtained from ordinary loops in
independent_oracle.py. Passing certifies the selected bank's stated properties,
not the sampler's entire proposal stream or a neural interpretation.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import platform
import sys
import time

import numpy as np

import independent_oracle as oracle

HERE = Path(__file__).resolve().parent
CONTRACT_SHA256 = "4b9dc2df3002dbbe56cbcc5d38c119ace801f16e2f3bd018d345deb4eda7af8c"
SOURCE_MAPPING_SHA256 = "5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d"
ENCODER_SHA256 = "5f0f54e797ffadeb7b802fec07eae425a877ce7cfed4a5c9e9353ca91a939c61"
PRIOR_FILE_SHA256 = {
    "sat_case/pilot_bank.npz": "610d02cc85ba825b7ac3100efd30ad1ec8f90f02e11a9fe8444f0c62a9e2971b",
    "sat_mechanism/query_tokens.npz": "208fe4a7e5ead1c007fdff57424ac7c5838c2f0c5be4272f95c8b49ad2686b62",
    "sat_case/candidate_logits.npz": "adad7106407676edc221165f1039c0477aca65ae1cd082c3bff08fb281da149d",
    "sat_case/selected_decision_check.npz": "7b00b522827574de5e604e6e650ad0ef8edf81e42b32a5ca7115c297edd9b942",
    "sat_case/primary_witness.npz": "cdc4ed22391542e406edb7fbe99f58da207cba3b69701b32804658753d83a152",
}
PRIOR_SOURCES = (
    ("sat_case/pilot_bank.npz", "tokens", "tokens", 4096),
    ("sat_mechanism/query_tokens.npz", "unique_tokens", "tokens", 4322),
    ("sat_case/candidate_logits.npz", "edited_clauses", "clause_ids", 256),
    ("sat_case/selected_decision_check.npz", "edited_clauses", "clause_ids", 768),
    ("sat_case/primary_witness.npz", "original_tokens", "tokens", 2),
    ("sat_case/primary_witness.npz", "edited_tokens", "tokens", 2),
)
COHORT_ORDER = ["positive", "negative"]
STATE_ORDER = ["original", "edited", "renamed_original", "renamed_edited"]


class VerificationError(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(str(receipt.get("failure")))


class _Audit:
    def __init__(self):
        self.started = time.monotonic()
        self.inputs = {}
        self.checks = Counter()
        self.where = {}

    def identity(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        value = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        self.inputs[str(path)] = value
        return value

    def require(self, name, passed, **details):
        if not passed:
            receipt = self.receipt("FAIL")
            receipt["failure"] = {"check": name, "location": self.where, **details}
            raise VerificationError(receipt)
        self.checks[name] += 1

    def receipt(self, status):
        return {"status": status,
                "scope": "Independent Boolean-loop selected-bank semantics and complete known-prior-input exclusion",
                "contract_sha256": CONTRACT_SHA256,
                "python": platform.python_version(), "numpy": np.__version__,
                "elapsed_seconds": time.monotonic() - self.started,
                "inputs": self.inputs, "passed_check_counts": dict(sorted(self.checks.items())),
                "new_formulas_generated": 0, "neural_forwards": 0, "fits": 0,
                "scientific_bootstrap_draws": 0,
                "limits": ["Does not independently regenerate the proposal stream or hash-selection ordering",
                           "Freshness is relative to the frozen known campaign ledger, not checkpoint training data",
                           "Semantic source-label tables are not claims about native activation correctness"]}


def _json(audit, path):
    audit.identity(path)
    return json.loads(Path(path).read_text())


def _array(audit, store, key, shape, allow_bool=False):
    audit.require("array_key_present", key in store, key=key)
    value = store[key]
    audit.require("array_shape", tuple(value.shape) == tuple(shape), key=key,
                  expected=list(shape), observed=list(value.shape))
    audit.require("integer_array_storage", value.dtype.kind in ("biu" if allow_bool else "iu"),
                  key=key, dtype=str(value.dtype))
    return value


def _verify_prior_ledger(audit, work_root, ledger_path, metadata_path):
    metadata = _json(audit, metadata_path)
    audit.identity(ledger_path)
    sources = metadata.get("sources", [])
    observed = [(entry.get("path"), entry.get("key"), entry.get("kind"), entry.get("row_count")) for entry in sources]
    audit.require("complete_six_source_ledger_inventory", len(observed) == 6 and set(observed) == set(PRIOR_SOURCES))
    audit.require("encoder_ledger_identity", metadata.get("encoder") ==
                  {"path": "sat_case/pilot_common.py", "sha256": ENCODER_SHA256})
    audit.require("unchanged_original_encoder", audit.identity(work_root / "sat_case/pilot_common.py")["sha256"] == ENCODER_SHA256)
    reconstructed = set()
    source_rows = 0
    for path, key, kind, row_count in PRIOR_SOURCES:
        audit.where = {"prior_source": path, "key": key}
        source_path = work_root / path
        actual_hash = audit.identity(source_path)["sha256"]
        entry = next(e for e in sources if e["path"] == path and e["key"] == key)
        audit.require("prior_source_pinned_identity", actual_hash == PRIOR_FILE_SHA256[path] == entry["sha256"])
        with np.load(source_path, allow_pickle=False) as source:
            rows = _array(audit, source, key, (row_count, 41 if kind == "tokens" else 10))
            paired_clause_key = {"tokens": "clauses", "original_tokens": "original_clauses",
                                 "edited_tokens": "edited_clauses"}.get(key)
            paired_clauses = (_array(audit, source, paired_clause_key, (row_count, 10))
                              if paired_clause_key is not None else None)
            for row_index, row in enumerate(rows):
                audit.where = {"prior_source": path, "key": key, "row": row_index}
                if kind == "tokens":
                    tokens = tuple(int(token) for token in row)
                    decoded = oracle.clause_ids_from_tokens(tokens)
                    audit.require("prior_token_roundtrip", oracle.tokens_from_clause_ids(decoded) == tokens)
                    if paired_clauses is not None:
                        audit.require("prior_saved_clause_token_agreement", decoded ==
                                      tuple(int(value) for value in paired_clauses[row_index]))
                else:
                    clause_ids = tuple(int(cid) for cid in row)
                    tokens = oracle.tokens_from_clause_ids(clause_ids)
                    audit.require("prior_clause_conversion", oracle.clause_ids_from_tokens(tokens) == clause_ids)
                reconstructed.add(oracle.token_bytes(tokens))
                source_rows += 1
    with np.load(ledger_path, allow_pickle=False) as saved:
        audit.require("prior_ledger_token_key", "tokens" in saved)
        rows = _array(audit, saved, "tokens", (len(reconstructed), 41))
        encoded = []
        for row_index, row in enumerate(rows):
            audit.where = {"prior_ledger_row": row_index}
            encoded.append(oracle.token_bytes(row))
        audit.require("prior_ledger_exact_sorted_unique_union", encoded == sorted(reconstructed))
    return reconstructed, {"source_rows": source_rows, "unique_rows": len(reconstructed), "source_entries": 6, "source_files": 5}


def _verify(audit, bank_path, metadata_path, ledger_path, ledger_metadata_path, work_root):
    audit.identity(HERE / "independent_oracle.py")
    audit.identity(HERE / "verify_bank.py")
    audit.require("pinned_execution_contract", audit.identity(work_root / "next_phase/sat_renaming_execution_contract.md")["sha256"] == CONTRACT_SHA256)
    mapping_path = work_root / "sat_mechanism/source_mapping.json"
    audit.require("pinned_source_mapping", audit.identity(mapping_path)["sha256"] == SOURCE_MAPPING_SHA256)
    mapping = json.loads(mapping_path.read_text())
    for unit, text in zip(oracle.UNIT_ORDER, oracle.ASSIGNMENT_TEXT):
        records = [record for record in mapping["rules"] if record["unit"] == unit]
        audit.require("source_singleton_assignment", len(records) == 1 and
                      records[0]["general_rule_is_single_assignment"] and
                      records[0]["general_rule_assignments"] == [text] and
                      records[0]["general_rule_ast"] == {"op": "phi", "assignment": text, "truth_table_index": oracle.assignment_index(text)}, unit=unit)
    prior, prior_summary = _verify_prior_ledger(audit, work_root, ledger_path, ledger_metadata_path)
    audit.where = {"stage": "bank_schema"}
    metadata = _json(audit, metadata_path)
    audit.identity(bank_path)
    audit.require("bank_contract_binding", metadata.get("contract_sha256") == CONTRACT_SHA256)
    audit.require("declared_axis_orders", metadata.get("unit_order") == list(oracle.UNIT_ORDER)
                  and metadata.get("cohort_order") == COHORT_ORDER and metadata.get("state_order") == STATE_ORDER)
    blocks = metadata.get("blocks", [])
    audit.require("64_matched_blocks", len(blocks) == 64)
    with np.load(bank_path, allow_pickle=False) as saved:
        clauses = _array(audit, saved, "clauses", (64, 2, 4, 10))
        tokens = _array(audit, saved, "tokens", (64, 2, 4, 41))
        counts = _array(audit, saved, "counts", (64, 2, 4, 32))
        labels = _array(audit, saved, "labels", (64, 2, 4))
        pairs = _array(audit, saved, "pair_index", (64,))
        trials = _array(audit, saved, "trial_ids", (64, 2))
        expected_bits = _array(audit, saved, "expected_unit_bits", (64, 2, 4, 4), allow_bool=True)
    endpoints, orbits, trial_set = set(), set(), set()
    descriptors = [set(), set()]
    pair_counts = Counter()
    task_labels = Counter()
    primary_positive = [0, 0]
    primary_negative = [0, 0]
    offpair_negative = [0, 0]
    slot_histograms = [Counter(), Counter()]
    source_assignments = tuple(oracle.assignment_index(text) for text in oracle.ASSIGNMENT_TEXT)
    for block_index, block in enumerate(blocks):
        audit.where = {"block": block_index}
        pair_index = int(pairs[block_index])
        audit.require("pair_round_robin", pair_index == block_index % 2)
        audit.require("block_metadata_identity", block.get("block_id") == block_index
                      and block.get("pair_index") == pair_index and block.get("pair") == "P" + str(pair_index + 1)
                      and block.get("trial_ids") == trials[block_index].tolist())
        audit.require("block_identity_vector_shapes", len(block.get("orbit_ids", [])) == 2
                      and len(block.get("endpoint_sha256", [])) == 2
                      and all(len(row) == 4 for row in block["endpoint_sha256"]))
        pair_counts[pair_index] += 1
        action = oracle.normalized_action(block["action"])
        renamed_action = oracle.rename_action(action)
        audit.require("action_rename_involution", oracle.rename_action(renamed_action) == action)
        descriptor = tuple(action[name] for name in oracle.ACTION_FIELDS)
        audit.require("descriptor_unique_within_pair", descriptor not in descriptors[pair_index])
        descriptors[pair_index].add(descriptor)
        slot_histograms[pair_index][action["slot"]] += 1
        anchor, partner = source_assignments[2 * pair_index:2 * pair_index + 2]
        audit.require("source_assignment_rename", oracle.rename_assignment(anchor) == partner and oracle.rename_assignment(partner) == anchor)
        a_values, b_values = oracle.assignment_bits(anchor), oracle.assignment_bits(partner)
        old_clause = oracle.clause_from_id(action["old_clause_id"], True)
        new_clause = oracle.clause_from_id(action["new_clause_id"], True)
        kept = action["kept_literal"]
        new_literal = next(token for token in new_clause if token != kept)
        audit.require("retained_literal_and_repair_orientation", not oracle.literal_value(kept, a_values)
                      and oracle.literal_value(kept, b_values) and not oracle.clause_value(old_clause, a_values)
                      and oracle.literal_value(new_literal, a_values) and oracle.clause_value(new_clause, b_values))
        for cohort in range(2):
            audit.where = {"block": block_index, "cohort": COHORT_ORDER[cohort]}
            trial = int(trials[block_index, cohort])
            audit.require("distinct_in_budget_trial", 0 <= trial < 100000 and trial not in trial_set)
            trial_set.add(trial)
            rows = [tuple(int(value) for value in tokens[block_index, cohort, state]) for state in range(4)]
            original, edited, renamed_original, renamed_edited = rows
            orbit = oracle.orbit_bytes(original)
            audit.require("unique_origin_renaming_orbit", orbit not in orbits)
            orbits.add(orbit)
            audit.require("stored_orbit_identity", block["orbit_ids"][cohort] == hashlib.sha256(orbit).hexdigest())
            audit.require("same_complete_action_both_cohorts", oracle.apply_action(original, action) == edited)
            audit.require("explicit_renaming_and_action_commutation", oracle.rename_tokens(original, True) == renamed_original
                          and oracle.rename_tokens(edited, True) == renamed_edited
                          and oracle.apply_action(renamed_original, renamed_action) == renamed_edited)
            results = []
            for state, row in enumerate(rows):
                audit.where = {"block": block_index, "cohort": COHORT_ORDER[cohort], "state": STATE_ORDER[state]}
                decoded = oracle.decode_tokens(row, restricted=True)
                audit.require("selected_clause_encoding", oracle.clause_ids_from_tokens(row, True) ==
                              tuple(int(value) for value in clauses[block_index, cohort, state]))
                audit.require("ten_slots_and_literal_positions_preserved", len(decoded) == 10 and
                              all(tuple(oracle.rename_literal(token) for token in clause) == oracle.rename_clause(clause, True) for clause in decoded))
                renamed = oracle.rename_tokens(row, True)
                audit.require("token_renaming_involution", oracle.rename_tokens(renamed, True) == row)
                row_bytes = oracle.token_bytes(row)
                audit.require("globally_unique_endpoint", row_bytes not in endpoints)
                endpoints.add(row_bytes)
                audit.require("endpoint_and_rename_excluded_from_prior_ledger", row_bytes not in prior and oracle.token_bytes(renamed) not in prior)
                audit.require("stored_endpoint_identity", block["endpoint_sha256"][cohort][state] == hashlib.sha256(row_bytes).hexdigest())
                result = oracle.evaluate_tokens(row, restricted=True)
                results.append(result)
                audit.require("all_32_semantic_counts", result["counts"] == tuple(int(value) for value in counts[block_index, cohort, state]))
                audit.require("truth_count_consistency", all(truth == (count == 0) for truth, count in zip(result["truth_table"], result["counts"])))
                audit.require("full_semantic_task_label", result["label"] == int(labels[block_index, cohort, state]))
                task_labels[result["label"]] += 1
                bit_table = tuple(int(result["truth_table"][index]) for index in source_assignments)
                audit.require("four_source_assignment_bit_labels", bit_table == tuple(int(value) for value in expected_bits[block_index, cohort, state]))
                target = [0, 0, 0, 0]
                if cohort == 0 and state in (1, 3):
                    target[2 * pair_index + (state == 3)] = 1
                audit.require("exact_frozen_four_unit_pattern", bit_table == tuple(target))
                for unit_index, bit in enumerate(bit_table):
                    if unit_index // 2 == pair_index:
                        primary_positive[pair_index] += bit
                        primary_negative[pair_index] += 1 - bit
                    else:
                        audit.require("offpair_semantic_bits_false", bit == 0)
                        offpair_negative[pair_index] += 1
            audit.where = {"block": block_index, "cohort": COHORT_ORDER[cohort]}
            expected_solutions = [(), (anchor,), (), (partner,)] if cohort == 0 else [(), (), (), ()]
            audit.require("all_endpoint_exact_solution_sets", [result["solutions"] for result in results] == expected_solutions)
            q_original = results[0]["counts"]
            q_edited = results[1]["counts"]
            audit.require("cohort_count_eligibility", (q_original[anchor] == 1 if cohort == 0 else q_original[anchor] >= 2)
                          and q_original[partner] >= 2)
            if cohort == 0:
                falsified = [slot for slot, clause in enumerate(oracle.decode_tokens(original, True)) if not oracle.clause_value(clause, a_values)]
                audit.require("positive_unique_false_occurrence_is_action_slot", falsified == [action["slot"]])
            for assignment in range(32):
                values = oracle.assignment_bits(assignment)
                update = int(not oracle.clause_value(old_clause, values)) - int(not oracle.clause_value(new_clause, values))
                audit.require("all_32_exact_count_updates", q_edited[assignment] == q_original[assignment] - update)
                renamed_assignment = oracle.rename_assignment(assignment)
                audit.require("all_32_renamed_counts_original", results[2]["counts"][renamed_assignment] == q_original[assignment])
                audit.require("all_32_renamed_counts_edited", results[3]["counts"][renamed_assignment] == q_edited[assignment])
            audit.require("anchor_decrements_partner_unchanged", q_edited[anchor] == q_original[anchor] - 1
                          and q_edited[partner] == q_original[partner])
    audit.where = {"stage": "global_totals"}
    audit.require("global_128_orbits_512_endpoints", len(orbits) == 128 and len(endpoints) == 512)
    audit.require("32_blocks_and_descriptors_per_pair", pair_counts == {0: 32, 1: 32} and [len(x) for x in descriptors] == [32, 32])
    audit.require("fixed_bit_denominators", primary_positive == [64, 64] and primary_negative == [448, 448]
                  and offpair_negative == [512, 512])
    audit.require("fixed_task_denominators", task_labels == {13: 128, 14: 384})
    result = audit.receipt("PASS")
    result["verified"] = {"matched_blocks": 64, "origin_orbits": 128, "endpoint_rows": 512,
                          "valuation_counts": 512 * 32, "unit_assignment_bits": 512 * 4,
                          "task_label_counts": dict(task_labels), "primary_positive_bits_per_pair": primary_positive,
                          "primary_negative_bits_per_pair": primary_negative, "offpair_negative_bits_per_pair": offpair_negative,
                          "descriptors_per_pair": [len(x) for x in descriptors],
                          "slot_histograms": [dict(sorted(c.items())) for c in slot_histograms], "prior_ledger": prior_summary}
    return result


def verify_bank(bank_path, metadata_path, ledger_path, ledger_metadata_path, work_root=None):
    audit = _Audit()
    work_root = Path(work_root).resolve() if work_root is not None else HERE.parent
    try:
        return _verify(audit, Path(bank_path), Path(metadata_path), Path(ledger_path), Path(ledger_metadata_path), work_root)
    except VerificationError:
        raise
    except Exception as exc:
        receipt = audit.receipt("FAIL")
        receipt["failure"] = {"check": "input_schema_or_oracle_exception", "location": audit.where,
                              "exception_type": type(exc).__name__, "message": str(exc)}
        raise VerificationError(receipt) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=HERE / "selected_bank.npz")
    parser.add_argument("--metadata", type=Path, default=HERE / "selected_bank.json")
    parser.add_argument("--ledger", type=Path, default=HERE / "prior_ledger_tokens.npz")
    parser.add_argument("--ledger-metadata", type=Path, default=HERE / "prior_ledger.json")
    parser.add_argument("--work-root", type=Path, default=HERE.parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = verify_bank(args.bank, args.metadata, args.ledger, args.ledger_metadata, args.work_root)
    except VerificationError as exc:
        receipt = exc.receipt
    if args.output:
        with args.output.open("x") as handle:
            handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "failure": receipt.get("failure"),
                      "verified": receipt.get("verified"), "output": str(args.output) if args.output else None}, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
