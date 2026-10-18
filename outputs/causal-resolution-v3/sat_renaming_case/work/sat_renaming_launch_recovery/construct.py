"""One bounded semantic proposal stream. No Torch, checkpoint access or model outcomes."""
from collections import Counter, defaultdict
from protocol import *


class SupportShortfall(RuntimeError):
    pass


def eligible_actions(formula, counts, pair_index, positive):
    _, a, b = PAIRS[pair_index]
    choices = []
    allowed = set(map(int, ALLOWED))
    for slot, cid0 in enumerate(formula):
        cid = int(cid0)
        if CLAUSE_TRUTH[cid, a]: continue
        c = CLAUSES[cid]
        for kept0 in c:
            kept = int(kept0)
            if LITERAL_TRUTH[kept, a] or not LITERAL_TRUTH[kept, b]: continue
            for variable in range(5):
                if variable in (int(c[0]) % 5, int(c[1]) % 5): continue
                literal = variable if ASSIGNMENTS[a, variable] else variable + 5
                new = CLAUSE_LOOKUP[tuple(sorted((kept, literal), key=lambda t: t % 5))]
                if new not in allowed: continue
                after = counts - (~CLAUSE_TRUTH[cid]).astype(np.int16) + (~CLAUSE_TRUTH[new]).astype(np.int16)
                solutions = np.flatnonzero(after == 0)
                if (positive and not np.array_equal(solutions, [a])) or (not positive and len(solutions)): continue
                if after[b] != counts[b] or after[a] != counts[a] - 1: raise RuntimeError("Edit-count algebra failed")
                choices.append((slot, cid, kept, variable, new))
    return choices


def main():
    frozen(); require_analysis_numpy()
    start, cpu = time.monotonic(), time.process_time()
    save_json("construction_started.json", {"utc": utc(), "freeze_sha256": sha(ROOT / "freeze.json"), "proposal_ceiling": PROPOSALS})
    diagnostics = Counter(); pair_diagnostics = [Counter(), Counter()]
    eligibility = np.zeros((PROPOSALS, 2), dtype=np.int8)
    action_multiplicity = np.zeros((PROPOSALS, 2), dtype=np.int16)
    records = []
    # One fixed vectorized draw; dtype and shape are part of the frozen stream.
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    candidate_clauses = ALLOWED[rng.integers(0, 36, size=(PROPOSALS, 10), dtype=np.int16)]
    candidate_counts = semantic_counts(candidate_clauses)
    save_npz("candidate_bank.npz", clauses=candidate_clauses, counts=candidate_counts)
    seen_originals = set()
    processed = 0
    try:
        for trial, (formula, counts) in enumerate(zip(candidate_clauses, candidate_counts)):
            if trial % 100 == 0: deadline(start, CONSTRUCTION_SECONDS, "construction")
            processed = trial + 1
            key = bytes(map(int, formula))
            if key in seen_originals:
                diagnostics["duplicate_original"] += 1; eligibility[trial, :] = 1; continue
            seen_originals.add(key)
            if np.any(counts == 0):
                diagnostics["initially_sat"] += 1; eligibility[trial, :] = 2; continue
            diagnostics["unique_initially_unsat"] += 1
            orb = orbit_bytes(formula)
            for pair_index, (_, a, b) in enumerate(PAIRS):
                pd = pair_diagnostics[pair_index]
                if counts[b] < 2:
                    pd["partner_count_below_two"] += 1; eligibility[trial, pair_index] = 3; continue
                positive = counts[a] == 1
                choices = eligible_actions(formula, counts, pair_index, positive)
                action_multiplicity[trial, pair_index] = len(choices)
                if not choices:
                    pd["no_eligible_action_positive" if positive else "no_eligible_action_negative"] += 1
                    eligibility[trial, pair_index] = 4; continue
                action = min(choices, key=lambda d: (digest("sat-renaming/action-v1", bytes([pair_index]), orb, bytes(d)), d))
                cohort = 0 if positive else 1
                eligibility[trial, pair_index] = 5 + cohort
                pd["eligible_positive" if positive else "eligible_negative"] += 1
                records.append({"trial": trial, "pair": pair_index, "cohort": cohort, "action": action,
                                "multiplicity": len(choices), "orbit": orb})
        deadline(start, CONSTRUCTION_SECONDS, "construction")
        save_npz("candidate_selection_data.npz", eligibility_codes=eligibility, action_multiplicity=action_multiplicity,
                 eligible_trial=np.array([r["trial"] for r in records], dtype=np.int64),
                 eligible_pair=np.array([r["pair"] for r in records], dtype=np.int8),
                 eligible_cohort=np.array([r["cohort"] for r in records], dtype=np.int8),
                 eligible_action=np.array([r["action"] for r in records], dtype=np.int16).reshape(-1, 5))
        pools = [defaultdict(lambda: [[], []]), defaultdict(lambda: [[], []])]
        for row in records: pools[row["pair"]][row["action"]][row["cohort"]].append(row)
        with np.load(ROOT / "prior_ledger_tokens.npz", allow_pickle=False) as prior: prior_set = {token_bytes(t) for t in prior["tokens"]}
        descriptors = []
        for p in range(2):
            for d, cohorts in pools[p].items():
                for cohort in cohorts:
                    cohort.sort(key=lambda r: (digest("sat-renaming/origin-v1", bytes([p]), bytes(d), bytes([r["cohort"]]), r["orbit"], token_bytes(encode(candidate_clauses[r["trial"]][None, :])[0])), r["trial"]))
            available = [d for d, c in pools[p].items() if c[0] and c[1]]
            pair_diagnostics[p]["descriptors_any"] = len(pools[p]); pair_diagnostics[p]["descriptors_both_cohorts"] = len(available)
            descriptors.append(iter(sorted(available, key=lambda d: (digest("sat-renaming/descriptor-v1", bytes([p]), bytes(d)), d))))
        used_orbits, used_endpoints, selected = set(), set(), []
        exclusions = []

        def assess(row, extra_endpoints=(), extra_orbits=()):
            four = quartet(candidate_clauses[row["trial"]], row["action"])
            tokens = encode(four)
            identities = [token_bytes(t) for t in tokens]
            if len(set(identities)) != 4: return None, "within_quartet_collision"
            if row["orbit"] in used_orbits or row["orbit"] in extra_orbits: return None, "origin_orbit_collision"
            if any(t in prior_set for t in identities): return None, "historical_input_overlap"
            if any(t in used_endpoints or t in extra_endpoints for t in identities): return None, "selected_endpoint_overlap"
            return (four, tokens, identities), None

        for round_index in range(QUOTA):
            for p in range(2):
                deadline(start, CONSTRUCTION_SECONDS, "construction")
                accepted = False
                for descriptor in descriptors[p]:
                    positive = pools[p][descriptor][0][0]
                    pos, reason = assess(positive)
                    if reason:
                        exclusions.append({"pair": p, "action": action_dict(descriptor), "positive_trial": positive["trial"], "reason": reason}); continue
                    negative = None; neg = None
                    for candidate in pools[p][descriptor][1]:
                        checked, reason = assess(candidate, pos[2], [positive["orbit"]])
                        if reason:
                            exclusions.append({"pair": p, "action": action_dict(descriptor), "negative_trial": candidate["trial"], "reason": reason}); continue
                        negative, neg = candidate, checked; break
                    if negative is None:
                        exclusions.append({"pair": p, "action": action_dict(descriptor), "positive_trial": positive["trial"], "reason": "first_positive_has_no_compatible_negative"}); continue
                    selected.append((p, descriptor, positive, negative, pos, neg))
                    used_orbits.update([positive["orbit"], negative["orbit"]]); used_endpoints.update(pos[2] + neg[2])
                    accepted = True; break
                if not accepted:
                    diagnostics["support_shortfall_pair_" + str(p)] += 1
                    raise SupportShortfall("Fixed proposal pool did not support 32 descriptors per pair")
        clauses = np.stack([np.stack([s[4][0], s[5][0]]) for s in selected])
        tokens = np.stack([np.stack([s[4][1], s[5][1]]) for s in selected])
        counts = semantic_counts(clauses.reshape(-1, 10)).reshape(64, 2, 4, 32)
        labels = np.where((counts == 0).any(axis=-1), 13, 14).astype(np.int16)
        expected = (counts[..., ASSIGNMENT_IDS] == 0).astype(np.int8)
        blocks = []
        for block_id, (p, d, pos, neg, pp, nn) in enumerate(selected):
            blocks.append({"block_id": block_id, "pair": PAIRS[p][0], "pair_index": p,
                           "trial_ids": [pos["trial"], neg["trial"]], "action": action_dict(d),
                           "orbit_ids": [hashlib.sha256(r["orbit"]).hexdigest() for r in [pos, neg]],
                           "endpoint_sha256": [[hashlib.sha256(t).hexdigest() for t in data[2]] for data in [pp, nn]],
                           "eligible_action_multiplicities": [pos["multiplicity"], neg["multiplicity"]]})
        save_npz("selected_bank.npz", clauses=clauses, tokens=tokens, counts=counts, labels=labels,
                 expected_unit_bits=expected, pair_index=np.array([s[0] for s in selected], dtype=np.int8),
                 trial_ids=np.array([[s[2]["trial"], s[3]["trial"]] for s in selected], dtype=np.int64))
        save_json("selected_bank.json", {"schema_version": 1, "contract_sha256": CONTRACT_SHA,
                  "implementation_freeze_sha256": sha(ROOT / "freeze.json"), "unit_order": UNITS.tolist(),
                  "cohort_order": COHORT_ORDER, "state_order": STATE_ORDER, "blocks": blocks,
                  "replay_block_ids": [0, 1], "target_population": "Equal-weight mixture of 32 selected action descriptors per pair under the frozen selection algorithm."})
        from verify_bank import verify_bank
        try:
            verification = verify_bank(ROOT / "selected_bank.npz", ROOT / "selected_bank.json", ROOT / "prior_ledger_tokens.npz", ROOT / "prior_ledger.json", work_root=WORK)
        except Exception as exc:
            save_json("verifier_failure.json", {"error": repr(exc), "receipt": getattr(exc, "receipt", None)}); raise
        save_json("bank_verification.json", verification)
        if verification.get("status") != "PASS": raise RuntimeError("Independent bank verifier did not return PASS")
        deadline(start, CONSTRUCTION_SECONDS, "construction")
        diagnostics["selected_blocks"] = len(selected)
        finish = {"status": "PASS", "utc": utc(), "proposals_drawn": PROPOSALS, "processed": processed,
                  "diagnostics": dict(diagnostics), "pair_diagnostics": [dict(c) for c in pair_diagnostics],
                  "eligibility_codebook": {"0": "not_processed", "1": "duplicate_original", "2": "initially_sat", "3": "partner_count_below_two", "4": "no_eligible_action", "5": "eligible_positive", "6": "eligible_negative"},
                  "selected_slot_counts": [dict(Counter(int(s[1][0]) for s in selected if s[0] == p)) for p in range(2)],
                  "selected_target_counts": [[candidate_counts[s[c]["trial"], PAIRS[p][1]].item() for s in selected if s[0] == p for c in [2, 3]] for p in range(2)],
                  "cost": cost(start, cpu)}
        save_json("selection_exclusions.json", exclusions)
        save_json("construction.json", finish)
        files = ["candidate_bank.npz", "candidate_selection_data.npz", "selected_bank.npz", "selected_bank.json", "bank_verification.json", "selection_exclusions.json", "construction.json"]
        save_json("selected_data_freeze.json", {"utc": utc(), "implementation_freeze_sha256": sha(ROOT / "freeze.json"), "files": [record(ROOT / f) for f in files]})
        selected_frozen()
    except Exception as exc:
        if not (ROOT / "candidate_selection_data.npz").exists():
            save_npz("candidate_selection_data.npz", eligibility_codes=eligibility, action_multiplicity=action_multiplicity)
        if "exclusions" in locals() and not (ROOT / "selection_exclusions.json").exists(): save_json("selection_exclusions.json", exclusions)
        partial = []
        if "selected" in locals():
            for block_id, (p, d, pos, neg, pp, nn) in enumerate(selected):
                partial.append({"block_id": block_id, "pair": PAIRS[p][0], "pair_index": p,
                    "action": action_dict(d), "trial_ids": [pos["trial"], neg["trial"]],
                    "orbit_ids": [hashlib.sha256(r["orbit"]).hexdigest() for r in [pos, neg]],
                    "endpoint_sha256": [[hashlib.sha256(t).hexdigest() for t in data[2]] for data in [pp, nn]],
                    "anchor_counts": [int(candidate_counts[r["trial"], PAIRS[p][1]]) for r in [pos, neg]],
                    "partner_counts": [int(candidate_counts[r["trial"], PAIRS[p][2]]) for r in [pos, neg]],
                    "eligible_action_multiplicities": [pos["multiplicity"], neg["multiplicity"]]})
        save_json("partial_selection.json", {"selected_so_far": partial,
                  "slot_counts": [dict(Counter(row["action"]["slot"] for row in partial if row["pair_index"]==p)) for p in range(2)],
                  "scope": "Preserved incomplete selection; not an admitted smaller native study."})
        if not (ROOT / "construction.json").exists():
            save_json("construction.json", {"status": "FEASIBILITY_STOP" if isinstance(exc, (SupportShortfall, TimeoutError)) else "ERROR_STOP", "utc": utc(), "error": repr(exc), "proposals_drawn": PROPOSALS, "processed": processed, "diagnostics": dict(diagnostics), "pair_diagnostics": [dict(c) for c in pair_diagnostics], "cost": cost(start, cpu)})
        if not isinstance(exc, (SupportShortfall, TimeoutError)): raise
    print(json.dumps(read_json("construction.json")))


if __name__ == "__main__": main()
