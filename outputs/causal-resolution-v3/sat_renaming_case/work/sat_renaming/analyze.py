"""Prespecified four-unit analysis, shared-variable numerical bounds and block bootstrap.

No fit, search, model prediction or row selection. RNG is used only inside main,
after the frozen bank, complete native extraction and numerical replay pass.
"""
import csv
from protocol import *


def radius(values):
    return ATOL + RTOL * np.abs(np.asarray(values, dtype=np.float64))


def bit_bounds(post):
    values = np.asarray(post, dtype=np.float64); r = radius(values)
    return (values >= .5).astype(np.int8), (values-r >= .5).astype(np.int8), (values+r >= .5).astype(np.int8)


def correctness(bits, lower, upper, targets):
    return np.stack([bits == targets, np.where(targets, lower, 1-upper), np.where(targets, upper, 1-lower)], axis=-1).astype(np.int8)


def enumerate_admissible(lower, upper):
    """All admissible bit assignments; deterministic enumeration, never random draws."""
    lo, hi = np.asarray(lower, dtype=np.int8).ravel(), np.asarray(upper, dtype=np.int8).ravel()
    if not np.all((lo >= 0) & (hi <= 1) & (lo <= hi)): raise ValueError("Invalid bit ranges")
    ambiguous = np.flatnonzero(lo != hi)
    n = 1 << len(ambiguous)
    out = np.broadcast_to(lo, (n, len(lo))).copy()
    if len(ambiguous): out[:, ambiguous] = ((np.arange(n, dtype=np.uint32)[:, None] >> np.arange(len(ambiguous), dtype=np.uint32)) & 1).astype(np.int8)
    return out


def triple(nominal, possibilities):
    return np.stack([nominal, possibilities.min(axis=0), possibilities.max(axis=0)], axis=-1).astype(np.float64)


def endpoint_statistics(nominal, lower, upper, primary_indices):
    """Two four-bit vectors, <=256 combinations; contrasts share the same bits."""
    values = enumerate_admissible(lower, upper).reshape(-1, 2, 4)
    source, target = values[:, 0], values[:, 1]
    nsource, ntarget = nominal[0], nominal[1]
    errors = np.stack([np.mean(target != source[:, perm], axis=1) for perm in PERMUTATIONS.values()], axis=1)
    nerrors = np.array([np.mean(ntarget != nsource[perm]) for perm in PERMUTATIONS.values()])
    contrasts = errors[:, 1:] - errors[:, :1]
    ncontrasts = nerrors[1:] - nerrors[0]
    perm = PERMUTATIONS["correct"]
    primary = np.sum(target[:, primary_indices] != source[:, perm[primary_indices]], axis=1)
    nprimary = np.sum(ntarget[primary_indices] != nsource[perm[primary_indices]])
    return triple(nerrors, errors), triple(ncontrasts, contrasts), triple(nprimary, primary)


def change_statistics(nominal, lower, upper, primary_indices):
    """Four four-bit vectors, <=65536 combinations, preserving shared coordinates."""
    values = enumerate_admissible(lower, upper).reshape(-1, 4, 4)
    original = values[:, 1] - values[:, 0]
    renamed = values[:, 3] - values[:, 2]
    no, nr = nominal[1]-nominal[0], nominal[3]-nominal[2]
    discrepancy = np.stack([np.mean(np.abs(renamed-original[:, perm]), axis=1) for perm in PERMUTATIONS.values()], axis=1)
    ndiscrepancy = np.array([np.mean(np.abs(nr-no[perm])) for perm in PERMUTATIONS.values()])
    primary = np.sum(renamed[:, primary_indices] != original[:, PERMUTATIONS["correct"][primary_indices]], axis=1)
    nprimary = np.sum(nr[primary_indices] != no[PERMUTATIONS["correct"][primary_indices]])
    return triple(ndiscrepancy, discrepancy), triple(nprimary, primary)


def task_bounds(logits, labels):
    values = np.asarray(logits, dtype=np.float64); r = radius(values)
    lo, hi = values-r, values+r
    possible = hi >= lo.max(axis=-1, keepdims=True)
    nominal = values.argmax(axis=-1)
    true_possible = np.take_along_axis(possible, np.asarray(labels)[..., None], axis=-1)[..., 0]
    guaranteed = (possible.sum(axis=-1) == 1) & true_possible
    good = np.stack([nominal == labels, guaranteed, true_possible], axis=-1).astype(np.int8)
    return nominal, possible, good


def minimum_condition(values, cutoff, strict=False):
    n, lo, hi = map(float, values)
    compare = (lambda x: x > cutoff) if strict else (lambda x: x >= cutoff)
    return {"values_nominal_lower_upper": [n, lo, hi], "comparison": ">" if strict else ">=", "cutoff": cutoff,
            "nominal_pass": bool(compare(n)), "adverse_pass": bool(compare(lo)), "favourable_pass": bool(compare(hi))}


def maximum_condition(values, cutoff):
    n, lo, hi = map(float, values)
    return {"values_nominal_lower_upper": [n, lo, hi], "comparison": "<=", "cutoff": cutoff,
            "nominal_pass": n <= cutoff, "adverse_pass": hi <= cutoff, "favourable_pass": lo <= cutoff}


def classify_conditions(conditions):
    values = list(conditions.values())
    if all(c["adverse_pass"] for c in values): return "PASS"
    if any(not c["favourable_pass"] for c in values): return "CRITERION_NOT_MET"
    return "UNRESOLVED_NUMERICAL_SENSITIVITY"


def affine_accounting(native, bank, class_possible, start):
    """Secondary only; no inference or hidden-unit search."""
    deadline(start, ANALYSIS_SECONDS, "analysis")
    with np.load(ROOT / "readout_weights.npz", allow_pickle=False) as data: weights = {k: data[k].astype(np.float64) for k in data.files}
    out, unembed, bias = weights["W_out"], weights["W_U"], weights["b_out"]
    if out.shape != (128, 512) or unembed.shape != (128, 16) or bias.shape != (128,): raise RuntimeError("Unexpected readout matrix shape")
    coefficients = out.T @ unembed
    reconstructed = native["resid_mid"].astype(np.float64) @ unembed + bias @ unembed + native["full_post"].astype(np.float64) @ coefficients
    residual = reconstructed-native["logits"].astype(np.float64)
    parity = bool(np.allclose(reconstructed, native["logits"], atol=ATOL, rtol=RTOL))
    save_npz("affine_reconstruction.npz", reconstructed_logits=reconstructed, residuals=residual)
    receipt = {"parity_pass": parity, "max_abs_residual": float(np.abs(residual).max()), "atol": ATOL, "rtol": RTOL,
               "scope": "Float64 affine reconstruction from saved float32 tensors; not patched native inference."}
    if not parity:
        receipt["status"] = "SECONDARY_PARITY_FAILED_NO_CLAMP_INTERPRETATION"
        save_json("affine_accounting.json", receipt); return receipt
    clamp = np.empty((64, 2, 2, 2, 16), dtype=np.float64)
    contributions = np.empty((64, 2, 2, 2), dtype=np.float64)
    for block in range(64):
        primary = np.arange(2*int(bank["pair_index"][block]), 2*int(bank["pair_index"][block])+2)
        off = np.array([u for u in range(4) if u not in primary])
        for direction, (before, after) in enumerate([(0, 1), (2, 3)]):
            for which, ids in enumerate([primary, off]):
                delta = np.take(native["post"][block, :, after].astype(np.float64) - native["post"][block, :, before].astype(np.float64), ids, axis=-1)
                coeff = coefficients[UNITS[ids]]
                change = delta @ coeff
                clamp[block, :, direction, which] = native["logits"][block, :, after].astype(np.float64)-change
                contributions[block, :, direction, which] = change[:, 13]-change[:, 14]
    edited_labels = bank["labels"][:, :, [1, 3]]
    clamp_labels = np.broadcast_to(edited_labels[..., None], (64, 2, 2, 2))
    pred, possible, good = task_bounds(clamp, clamp_labels)
    native_pred = native["logits"][:, :, [1, 3]].argmax(axis=-1)
    native_unique = class_possible[:, :, [1, 3]].sum(axis=-1) == 1
    stable_change = (native_unique[..., None] & (possible.sum(axis=-1) == 1) & (pred != native_pred[..., None]))
    nominal_change = pred != native_pred[..., None]
    ordered = np.sort(clamp, axis=-1)
    save_npz("affine_clamps.npz", logits=clamp, sat_minus_unsat_contribution=contributions,
             nominal_class=pred, possible_classes=possible, correctness_bounds=good,
             nominal_class_change=nominal_change, numerically_stable_class_change=stable_change,
             winning_gap=ordered[..., -1]-ordered[..., -2])
    summaries = {}
    for cohort, name in enumerate(COHORT_ORDER):
        for which, qname in enumerate(["nominated_pair", "other_fixed_pair"]):
            before_correct = native_pred[:, cohort] == edited_labels[:, cohort]
            after_correct = pred[:, cohort, :, which] == edited_labels[:, cohort]
            summaries[name+"/"+qname] = {"edited_endpoints": 128, "nominal_changes": int(nominal_change[:, cohort, :, which].sum()),
                "stable_changes": int(stable_change[:, cohort, :, which].sum()),
                "correct_to_wrong": int((before_correct & ~after_correct).sum()),
                "wrong_to_correct": int((~before_correct & after_correct).sum()),
                "clamp_nonanswer": int((~np.isin(pred[:, cohort, :, which], [13, 14])).sum()),
                "mean_signed_sat_minus_unsat_contribution": float(contributions[:, cohort, :, which].mean())}
    receipt.update(status="PASS_SECONDARY_ACCOUNTING_ONLY", summaries=summaries,
                   caveat="Downstream clamp with edited residual and all other postactivations fixed; potentially off-manifold, not upstream or unique causal routing.")
    save_json("affine_accounting.json", receipt); return receipt


def main():
    selected_frozen(); require_analysis_numpy()
    initial, replay = read_json("native_initial.json"), read_json("native_replay.json")
    if initial["status"] != "PASS" or replay["status"] != "PASS": raise RuntimeError("Native/replay gate failed; no scientific draws allowed")
    for filename, receipt in [("native_outputs.npz", initial), ("replay_outputs.npz", replay)]: require_hash(ROOT / filename, receipt["output_sha256"])
    require_hash(ROOT / "readout_weights.npz", initial["weights_sha256"])
    start, cpu = time.monotonic(), time.process_time()
    save_json("analysis_started.json", {"utc": utc(), "freeze_sha256": sha(ROOT / "freeze.json"), "bootstrap_draw_ceiling": BOOTSTRAP_DRAWS})
    with np.load(ROOT / "selected_bank.npz", allow_pickle=False) as data: bank = {k: data[k].copy() for k in data.files}
    with np.load(ROOT / "native_outputs.npz", allow_pickle=False) as data: native = {k: data[k].copy() for k in data.files}
    if not np.array_equal(native["units"], UNITS) or not np.array_equal(native["block_ids"], np.arange(64)): raise RuntimeError("Native/bank row alignment failure")
    bits, lower, upper = bit_bounds(native["post"])
    good = correctness(bits, lower, upper, bank["expected_unit_bits"])
    pred, possible_classes, task_good = task_bounds(native["logits"], bank["labels"])
    full_mapping = np.empty((64, 2, 3, 3)); edited_mapping = np.empty_like(full_mapping)
    full_contrasts = np.empty((64, 2, 2, 3)); edited_contrasts = np.empty_like(full_contrasts)
    changes = np.empty_like(full_mapping)
    primary_transport = np.empty((64, 2, 3)); primary_changes = np.empty_like(primary_transport)
    metrics, denominators = {}, {}
    def add(name, values, denominator):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (64, 3) or not np.all(values[:, 1] <= values[:, 0]+1e-12) or not np.all(values[:, 0] <= values[:, 2]+1e-12): raise RuntimeError("Invalid bounded metric: "+name)
        metrics[name] = values; denominators[name] = denominator
    primary_good = np.empty((64, 2, 4, 2, 3), dtype=np.int8)
    primary_expected = np.empty((64, 2, 4, 2), dtype=np.int8)
    off_good = np.empty_like(primary_good)
    for block in range(64):
        deadline(start, ANALYSIS_SECONDS, "analysis")
        ids = np.arange(2*int(bank["pair_index"][block]), 2*int(bank["pair_index"][block])+2)
        off = np.array([i for i in range(4) if i not in ids])
        primary_good[block] = np.take(good[block], ids, axis=2)
        primary_expected[block] = np.take(bank["expected_unit_bits"][block], ids, axis=2)
        off_good[block] = np.take(good[block], off, axis=2)
        for cohort in range(2):
            endpoints = []
            for state in [0, 1]:
                states = [state, state+2]
                endpoints.append(endpoint_statistics(bits[block, cohort, states], lower[block, cohort, states], upper[block, cohort, states], ids))
            full_mapping[block, cohort] = (endpoints[0][0]+endpoints[1][0])/2
            edited_mapping[block, cohort] = endpoints[1][0]
            full_contrasts[block, cohort] = (endpoints[0][1]+endpoints[1][1])/2
            edited_contrasts[block, cohort] = endpoints[1][1]
            primary_transport[block, cohort] = endpoints[0][2]+endpoints[1][2]
            changes[block, cohort], primary_changes[block, cohort] = change_statistics(bits[block, cohort], lower[block, cohort], upper[block, cohort], ids)
    posmask = primary_expected[..., None] == 1
    negmask = ~posmask
    add("primary_sensitivity", (primary_good*posmask).sum(axis=(1,2,3)), 2)
    add("primary_specificity", (primary_good*negmask).sum(axis=(1,2,3)), 14)
    add("primary_balanced_accuracy", ((primary_good*posmask).sum(axis=(1,2,3))/2 + (primary_good*negmask).sum(axis=(1,2,3))/14)/2, 1)
    positive_pattern = primary_good[:, 0].all(axis=(1,2)).astype(np.int8)
    negative_pattern = primary_good[:, 1].all(axis=(1,2)).astype(np.int8)
    add("positive_exact_pattern", positive_pattern, 1); add("negative_exact_pattern", negative_pattern, 1)
    off_tn = off_good.sum(axis=(1,2,3))
    # FPR bounds invert TN correctness bounds; adverse upper FP uses lower TN.
    add("off_pair_false_positive_rate", np.stack([16-off_tn[:,0], 16-off_tn[:,2], 16-off_tn[:,1]], axis=-1), 16)
    for name, states in [("original", [0,1]), ("renamed", [2,3])]:
        g = primary_good[:, :, states]; t = primary_expected[:, :, states, :, None]
        add(name+"_sensitivity", (g*(t==1)).sum(axis=(1,2,3)), 1)
        add(name+"_specificity", (g*(t==0)).sum(axis=(1,2,3)), 7)
    add("primary_transport_mismatch_rate", primary_transport.sum(axis=1), 8)
    add("primary_response_change_mismatch_rate", primary_changes.sum(axis=1), 4)
    for mi, mapping in enumerate(PERMUTATIONS):
        for cohort, name in enumerate(COHORT_ORDER):
            add(mapping+"_"+name+"_endpoint_error", full_mapping[:,cohort,mi], 1)
            add(mapping+"_"+name+"_edited_endpoint_error", edited_mapping[:,cohort,mi], 1)
            add(mapping+"_"+name+"_signed_change_discrepancy", changes[:,cohort,mi], 1)
    for ci, comparator in enumerate(["identity", "wrong"]):
        add("edited_advantage_"+comparator, edited_contrasts[:,0,ci], 1)
        add("full_positive_advantage_"+comparator, full_contrasts[:,0,ci], 1)
    task_pos = (bank["labels"] == 13)[...,None]
    task_neg = (bank["labels"] == 14)[...,None]
    task_tp = (task_good*task_pos).sum(axis=(1,2)); task_tn = (task_good*task_neg).sum(axis=(1,2))
    add("task_sensitivity", task_tp, 2); add("task_specificity", task_tn, 6)
    add("task_balanced_accuracy", (task_tp/2+task_tn/6)/2, 1)
    add("task_accuracy", task_good.sum(axis=(1,2)), 8)
    for cohort, name in enumerate(COHORT_ORDER):
        quartet_good = task_good[:,cohort].all(axis=1).astype(np.int8)
        add("task_"+name+"_accuracy", task_good[:,cohort].sum(axis=1), 4)
        add("task_"+name+"_complete_quartet", quartet_good, 1)
        pattern = positive_pattern if cohort == 0 else negative_pattern
        add("joint_"+name+"_pattern_and_task", pattern*quartet_good, 1)
    add("task_positive_sensitivity", (task_good[:,0]*task_pos[:,0]).sum(axis=1), 2)
    add("task_positive_specificity", (task_good[:,0]*task_neg[:,0]).sum(axis=1), 2)
    # No SAT cases exist in controls: their sensitivity is explicitly undefined in JSON.
    add("task_negative_specificity", task_good[:,1].sum(axis=1), 4)
    names = list(metrics)
    values = np.stack([metrics[n] for n in names], axis=1)
    denoms = np.array([denominators[n] for n in names], dtype=np.float64)
    rates = values/denoms[None,:,None]
    pair_indices = [np.flatnonzero(bank["pair_index"]==p) for p in range(2)]
    point = np.stack([rates[idx].mean(axis=0) for idx in pair_indices])
    point = np.concatenate([point, point.mean(axis=0, keepdims=True)], axis=0)
    deadline(start, ANALYSIS_SECONDS, "analysis")
    # The only scientific bootstrap RNG call, after all upstream gates pass.
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = rng.integers(0, QUOTA, size=(BOOTSTRAP_DRAWS, 2, QUOTA), dtype=np.int16)
    save_npz("bootstrap_indices.npz", indices=draws)
    bootstrap = np.stack([rates[pair_indices[p][draws[:,p]]].mean(axis=1) for p in range(2)], axis=1)
    bootstrap = np.concatenate([bootstrap, bootstrap.mean(axis=1,keepdims=True)], axis=1)
    intervals = np.quantile(bootstrap, [.025,.975], axis=0, method="linear")
    if not np.isfinite(bootstrap).all(): raise RuntimeError("Missing statistic; bootstrap draws cannot be replenished")
    save_npz("bounded_metrics.npz", metric_names=np.asarray(names), denominators_per_block=denoms, block_values=values,
             point_rates=point, bootstrap_rates=bootstrap, interval_quantiles=intervals,
             scope_names=np.array(["P1","P2","pooled"]), bound_order=np.array(["nominal","lower","upper"]))
    save_npz("event_bounds.npz", nominal_bits=bits, lower_bits=lower, upper_bits=upper, correctness=good,
             nominal_task_class=pred, possible_task_classes=possible_classes, task_correctness=task_good,
             full_mapping=full_mapping, edited_mapping=edited_mapping, full_contrasts=full_contrasts,
             edited_contrasts=edited_contrasts, signed_change_discrepancy=changes,
             signed_changes=np.stack([bits[:,:,1]-bits[:,:,0],bits[:,:,3]-bits[:,:,2]],axis=2),
             primary_transport_mismatch=primary_transport, primary_response_change_mismatch=primary_changes)
    benchmark, directions = {}, {}
    for p, ids in enumerate(pair_indices):
        pair_name = PAIRS[p][0]
        for name, threshold, maximum in [("primary_sensitivity",58,False),("primary_specificity",404,False),("positive_exact_pattern",26,False),("off_pair_false_positive_rate",51,True)]:
            aggregate = metrics[name][ids].sum(axis=0)
            benchmark[pair_name+"/"+name] = maximum_condition(aggregate,threshold) if maximum else minimum_condition(aggregate,threshold)
        for comparator in ["identity","wrong"]:
            benchmark[pair_name+"/edited_advantage_"+comparator] = minimum_condition(metrics["edited_advantage_"+comparator][ids].mean(axis=0),0,strict=True)
        directions[pair_name] = {}
        for direction in ["original","renamed"]:
            conditions = {"true_positive": minimum_condition(metrics[direction+"_sensitivity"][ids].sum(axis=0),29),
                          "true_negative": minimum_condition(metrics[direction+"_specificity"][ids].sum(axis=0),202)}
            directions[pair_name][direction] = {"status":classify_conditions(conditions),"conditions":conditions,"positive_denominator":32,"negative_denominator":224}
        directions[pair_name]["localized_transport_limitation"] = directions[pair_name]["original"]["status"]=="PASS" and directions[pair_name]["renamed"]["status"]=="CRITERION_NOT_MET"
    for comparator in ["identity","wrong"]:
        index = names.index("edited_advantage_"+comparator)
        benchmark["pooled/interval_lower_"+comparator] = minimum_condition(intervals[0,2,index],0,strict=True)
    status = classify_conditions(benchmark)
    summaries = {}
    def denominator_metadata(name, denominator, blocks):
        result = {"aggregation_denominator":int(denominator*blocks), "matched_resampling_blocks":blocks,
                  "denominator_unit":"atomic target events" if denominator!=1 else "matched-block averages of normalized metrics or complete-origin indicators"}
        if name.endswith("_edited_endpoint_error") or name.startswith("edited_advantage_"):
            result["atomic_comparisons_per_mapping_per_block"] = 4
        elif name.endswith("_endpoint_error") or name.startswith("full_positive_advantage_"):
            result["atomic_comparisons_per_mapping_per_block"] = 8
        elif name.endswith("_signed_change_discrepancy"):
            result.update(atomic_comparisons_per_mapping_per_block=4, atomic_statistic="absolute signed-change discrepancy, range 0 to 2")
        elif "exact_pattern" in name:
            result.update(primary_bits_per_origin_pattern=8, complete_origin_indicators_per_block=1)
        elif "complete_quartet" in name:
            result.update(task_classes_per_complete_origin=4, complete_origin_indicators_per_block=1)
        elif name.startswith("joint_"):
            result.update(primary_bits_per_origin_pattern=8, task_classes_per_complete_origin=4, complete_origin_indicators_per_block=1)
        elif name=="primary_balanced_accuracy":
            result["class_event_denominators_per_block"]={"positive":2,"negative":14}
        elif name=="task_balanced_accuracy":
            result["class_event_denominators_per_block"]={"positive":2,"negative":6}
        return result
    for si, scope in enumerate(["P1","P2","pooled"]):
        summaries[scope] = {name:{"point_nominal_lower_upper":point[si,ni].tolist(),
                                  "bootstrap_025_nominal_lower_upper":intervals[0,si,ni].tolist(),
                                  "bootstrap_975_nominal_lower_upper":intervals[1,si,ni].tolist(),
                                  **denominator_metadata(name,denoms[ni],64 if si==2 else 32)} for ni,name in enumerate(names)}
    with (ROOT/"input_unit_outcomes.csv").open("x",newline="") as f:
        writer=csv.writer(f); writer.writerow(["block","pair","cohort","state","unit","role","pre","post","expected_bit","nominal_bit","lower_bit","upper_bit","oracle_class","native_class","possible_classes"])
        for block in range(64):
            p=int(bank["pair_index"][block])
            for cohort in range(2):
                for state in range(4):
                    for u in range(4):
                        writer.writerow([block,PAIRS[p][0],COHORT_ORDER[cohort],STATE_ORDER[state],int(UNITS[u]),"primary" if u//2==p else "off_pair",
                            float(native["pre"][block,cohort,state,u]),float(native["post"][block,cohort,state,u]),int(bank["expected_unit_bits"][block,cohort,state,u]),
                            int(bits[block,cohort,state,u]),int(lower[block,cohort,state,u]),int(upper[block,cohort,state,u]),int(bank["labels"][block,cohort,state]),int(pred[block,cohort,state]),
                            " ".join(map(str,np.flatnonzero(possible_classes[block,cohort,state])))])
    affine = affine_accounting(native, bank, possible_classes, start)
    deadline(start, ANALYSIS_SECONDS, "analysis")
    output = {"status":status,"utc":utc(),"contract_sha256":CONTRACT_SHA,"benchmark_conditions":benchmark,"directional_criteria":directions,
              "metrics":summaries,"nominal_benchmark_pass":all(c["nominal_pass"] for c in benchmark.values()),
              "ambiguous_bit_positions":int((lower!=upper).sum()),"ambiguous_task_rows":int((possible_classes.sum(axis=-1)>1).sum()),
              "native_nonanswer_rows":int((~np.isin(pred,[13,14])).sum()),"negative_control_task_sensitivity":None,
              "bootstrap":{"draws":BOOTSTRAP_DRAWS,"valid_draws":int(np.isfinite(bootstrap).all(axis=(1,2,3)).sum()),"seed":BOOTSTRAP_SEED,"quantile_method":"linear","unit":"matched resampling block, stratified by fixed pair","indices_sha256":sha(ROOT/"bootstrap_indices.npz")},
              "cost":cost(start,cpu),"native_cost":{"initial_input_forwards":initial["input_forwards"],"replay_input_forwards":replay["input_forwards"],"batch_calls":initial["batch_calls"]+replay["batch_calls"],"fits":0},
              "secondary_affine":affine,"target_population":"Equal-weight mixture of the 32 selected descriptors per pair under the frozen selection algorithm; one fixed checkpoint.",
              "scope":"Prospective new native responses in a follow-on chosen after earlier development. No unique-mechanism, readiness, generic-margin exclusion or generality claim."}
    save_json("analysis.json",output)
    report=["# SAT renaming: bounded source-fidelity result", "", "Status: **"+status+"**.", "", output["target_population"],
            "The benchmark is an author-chosen finite-cohort criterion; it is not a guarantee from the source paper. The two fixed pairs are not a sample of a unit population.", "",
            "| Pair | Primary sensitivity | Primary specificity | Exact positive pattern | Off-pair false-positive rate | Identity advantage | Wrong-map advantage |", "|---|---:|---:|---:|---:|---:|---:|"]
    for si,name in enumerate(["P1","P2"]):
        chosen=["primary_sensitivity","primary_specificity","positive_exact_pattern","off_pair_false_positive_rate","edited_advantage_identity","edited_advantage_wrong"]
        report.append("| "+name+" | "+" | ".join(f"{point[si,names.index(k),0]:.6f}" for k in chosen)+" |")
    report += ["", "All 512 task rows and 2,048 four-unit event rows remain in the analysis. Task performance and numerical sensitivity are reported separately in analysis.json and input_unit_outcomes.csv.",
               "", "Pooled bootstrap intervals are descriptive conditional intervals, not two separately supported interval claims. Primary bit patterns, transport errors and mapping advantages reuse the same events.",
               "", "A failed criterion limits this selected extension; it does not refute the source's distribution-dependent interpretation. A positive result remains compatible with assignment-selective scalar margins or confidence-plus-syntax models.",
               "", "Native cost: 512 initial and 16 replay input forwards in 66 eight-row batch calls; zero fits. Bootstrap: exactly 2,000 saved block draws, with the same indices used for nominal and numerical-bound statistics.",
               "", "Secondary affine accounting: "+affine["status"]+". It is a float64 downstream clamp calculation from saved native arrays, not a patched-native experiment or an upstream mechanism discovery.", ""]
    with (ROOT/"REPORT.md").open("x") as f:f.write("\n".join(report))
    print(json.dumps({"status":status,"analysis_sha256":sha(ROOT/"analysis.json"),"bootstrap_draws":BOOTSTRAP_DRAWS}))


if __name__ == "__main__": main()
