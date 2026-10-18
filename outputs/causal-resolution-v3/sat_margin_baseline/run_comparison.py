"""Exactly one frozen cached-data fit/assessment; never imports a neural runtime."""
import csv
import datetime
import os
import pickle
import resource
import time
import traceback
import warnings

from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits
from common import *


def csv_write(name, rows):
    with (ROOT/name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def cached_data():
    data = dict(np.load(MECH/"stage_b_raw.npz", allow_pickle=False))
    queries = read_json(MECH/"selected_queries.json")["rows"]
    rows = read_json(MECH/"source_mapping.json")["single_assignment_general_units"]
    assert [r["unit"] for r in rows] == UNITS
    n = len(queries)
    assert n == 2176 and np.array_equal(data["query_id"], np.arange(n))
    assert data["original_pre"].dtype == np.float32
    assert data["native_high"].shape == (n,2)
    assert np.array_equal(data["native_high"], data["native_edited_post"] >= .5)
    assert np.array_equal(data["clipped"], (data["original_q"] >= 1) & (data["original_pre"] < 0))
    assert np.array_equal(data["action_effect"], np.tile([1,0], (n,1)))
    assert np.isin(data["split"], ["fit", "assessment"]).all()
    assert np.array_equal(data["split"] == "fit", data["origin_index"] < 3072)
    assert ((data["origin_index"] >= 0) & (data["origin_index"] < 4096)).all()
    # Check the declared action feature from the canonical grammar, without a formula oracle.
    clauses = np.array([(i+5*a,j+5*b) for i in range(5) for j in range(i+1,5)
                        for a in range(2) for b in range(2)])
    assignments = ((np.arange(32)[:,None] >> np.arange(4,-1,-1)) & 1).astype(bool)
    literal_truth = np.concatenate((assignments,~assignments), axis=1).T
    clause_truth = literal_truth[clauses].any(axis=1)
    for k,q in enumerate(queries):
        for key, source in [("query_id","query_id"),("unit_index","unit_index"),
                            ("origin_index","origin_index"),("split","split"),("original_q","original_q")]:
            assert data[key][k] == q[source]
        assert q["unit"] == UNITS[q["unit_index"]]
        mapping = rows[q["unit_index"]]
        assert q["assignment"] == mapping["assignment"]
        assert q["assignment_index"] == mapping["truth_table_index"]
        expected_context = [[q["slot"],q["old_clause"],q["repair_clause"]],
                            [q["slot"],q["old_clause"],q["neutral_clause"]]]
        assert np.array_equal(data["context_rows"][k], expected_context)
        a = q["assignment_index"]
        effects = int(not clause_truth[q["old_clause"],a]) - (~clause_truth[[q["repair_clause"],q["neutral_clause"]],a]).astype(int)
        assert np.array_equal(effects, data["action_effect"][k])
    structured_index = list(data["model_names"]).index("structured")
    data["structured"] = data["predictions"][:,:,structured_index]
    assert np.isin(data["structured"], [0,1]).all()
    data["x"] = design_matrix(np.repeat(data["original_pre"],2),
                              data["action_effect"].reshape(-1), data["context_rows"].reshape(-1,3))
    return data, rows


def serialize_estimator(model, path):
    """Full estimator plus inspectable arrays needed by raw or binned prediction."""
    path.with_suffix(".pkl").write_bytes(pickle.dumps(model, protocol=5))
    arrays = dict(baseline_prediction=np.asarray(model._baseline_prediction),
                  classes=np.asarray(model.classes_),
                  n_bins_non_missing=np.asarray(model._bin_mapper.n_bins_non_missing_),
                  is_categorical=np.asarray(model._bin_mapper.is_categorical_))
    for j, thresholds in enumerate(model._bin_mapper.bin_thresholds_):
        arrays[f"bin_thresholds_{j:03d}"] = np.asarray(thresholds)
    trees = []
    for iteration, ensemble in enumerate(model._predictors):
        for within, tree in enumerate(ensemble):
            key = f"tree_{iteration:03d}_{within:02d}"
            arrays[key+"_nodes"] = tree.nodes
            arrays[key+"_raw_cat_bitsets"] = tree.raw_left_cat_bitsets
            arrays[key+"_binned_cat_bitsets"] = tree.binned_left_cat_bitsets
            leaves = int(tree.nodes["is_leaf"].sum())
            depth = int(tree.nodes["depth"].max())
            assert leaves <= 7 and depth <= 3
            trees.append(dict(key=key, nodes=len(tree.nodes), leaves=leaves,
                              internal_splits=len(tree.nodes)-leaves, depth=depth))
    assert model.n_features_in_ == 92 and model.n_iter_ == 200
    assert all(not value.dtype.hasobject for value in arrays.values())
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    detail = dict(parameters=model.get_params(), classes=model.classes_.astype(int).tolist(),
        n_features_in=int(model.n_features_in_), n_iter=int(model.n_iter_),
        n_trees_per_iteration=int(model.n_trees_per_iteration_),
        baseline_prediction=model._baseline_prediction.tolist(),
        missing_values_bin_idx=int(model._bin_mapper.missing_values_bin_idx_),
        bin_mapper_parameters={k:v for k,v in model._bin_mapper.get_params().items()
                               if k not in ["is_categorical", "known_categories"]},
        categorical_features=None, preprocessing="identity numeric columns; no categorical features",
        trees=trees, tree_count=len(trees), leaf_count=sum(t["leaves"] for t in trees),
        internal_split_count=sum(t["internal_splits"] for t in trees),
        maximum_depth=max(t["depth"] for t in trees),
        raw_prediction_arrays_bytes=sum(a.nbytes for a in arrays.values()),
        full_estimator_pickle_bytes=path.with_suffix(".pkl").stat().st_size,
        prediction_arrays_npz_bytes=path.with_suffix(".npz").stat().st_size,
        array_schema={k:dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes)
                      for k,a in arrays.items()})
    write_json(path.with_suffix(".json"), detail)
    return {k:detail[k] for k in ["tree_count", "leaf_count", "internal_split_count", "maximum_depth",
            "raw_prediction_arrays_bytes", "full_estimator_pickle_bytes", "prediction_arrays_npz_bytes"]}


def fit_once(data, rows, execution):
    n = len(data["query_id"])
    probability = np.full((n,2), np.nan, dtype=np.float64)
    prediction = probability.copy()
    sample_weights = probability.copy()
    all_records = []
    (ROOT/"models").mkdir(exist_ok=False)
    for ui,row in enumerate(rows):
        unit = row["unit"]
        fit_pair = (data["unit_index"] == ui) & (data["split"] == "fit") & data["clipped"]
        all_pair = data["unit_index"] == ui
        fit_action = np.repeat(fit_pair,2); all_action = np.repeat(all_pair,2)
        target = data["native_high"][fit_pair].reshape(-1)
        counts = [int((target == c).sum()) for c in [False, True]]
        record = dict(unit=unit, assignment=row["assignment"], fit_paired_origins=int(fit_pair.sum()),
                      fit_action_rows=len(target), negative=counts[0], positive=counts[1],
                      warnings=[], fit_seconds=0., prediction_seconds=0., fit_attempts=0,
                      parameters=estimator().get_params(), fitted_object=None)
        path = ROOT/"models"/f"unit_{unit:03d}"
        if not len(target):
            record.update(status="UNESTIMABLE_NO_FIT_ROWS", reason="No selected clipped fit action rows")
            write_json(path.with_suffix(".json"), record)
        elif not all(counts):
            value = bool(counts[1])
            probability[all_pair] = value; prediction[all_pair] = value
            record.update(status="CONSTANT_ONE_CLASS", value=value,
                          warnings=[dict(category="OneClassFit", message="Only one native fit class; constant retained")])
            write_json(path.with_suffix(".json"), record)
        else:
            weights = np.where(target, len(target)/(2*counts[1]), len(target)/(2*counts[0]))
            assert np.isclose(weights[target].sum(),len(target)/2)
            assert np.isclose(weights[~target].sum(),len(target)/2)
            sample_weights[fit_pair] = weights.reshape(-1,2)
            record["class_sample_weights"] = dict(negative=len(target)/(2*counts[0]), positive=len(target)/(2*counts[1]))
            model = estimator()
            execution["classifier_fit_calls"] += 1; record["fit_attempts"] = 1
            assert execution["classifier_fit_calls"] <= 17
            write_json(ROOT/"execution.json", execution)
            start = time.monotonic()
            try:
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    model.fit(data["x"][fit_action], target, sample_weight=weights)
                record["warnings"] = [dict(category=w.category.__name__,message=str(w.message)) for w in observed]
                record["status"] = "FITTED"
            except Exception as error:
                record.update(status="FIT_EXCEPTION", exception=repr(error), traceback=traceback.format_exc())
                record["warnings"] = [dict(category=w.category.__name__,message=str(w.message)) for w in observed]
            record["fit_seconds"] = time.monotonic()-start
            if record["status"] == "FITTED":
                record["serialization"] = serialize_estimator(model,path)
                record["fitted_object"] = str(path.with_suffix(".pkl").relative_to(ROOT))
                start = time.monotonic()
                try:
                    with warnings.catch_warnings(record=True) as observed:
                        warnings.simplefilter("always")
                        result = model.predict_proba(data["x"][all_action])
                    record["warnings"] += [dict(category=w.category.__name__,message=str(w.message)) for w in observed]
                    positive_class = int(np.flatnonzero(model.classes_ == True)[0])
                    raw_probability = result[:,positive_class]
                    np.savez_compressed(path.with_name(path.name+"_prediction_raw.npz"),
                        query_id=data["query_id"][all_pair], probabilities=result, classes=model.classes_)
                    if raw_probability.dtype != np.float64 or not np.isfinite(raw_probability).all() or not ((raw_probability >= 0) & (raw_probability <= 1)).all():
                        record["status"] = "NONFINITE_OR_INVALID_PROBABILITY"
                    else:
                        probability[all_pair] = raw_probability.reshape(-1,2)
                        prediction[all_pair] = (raw_probability > .5).reshape(-1,2)
                        record["probability_point5_ties"] = int((raw_probability == .5).sum())
                        record["status"] = "FITTED_AND_PREDICTED"
                except Exception as error:
                    record.update(status="PREDICTION_EXCEPTION", exception=repr(error), traceback=traceback.format_exc())
                record["prediction_seconds"] = time.monotonic()-start
        all_records.append(record)
        write_json(ROOT/"fit_records.json", all_records)
        print(json.dumps(dict(unit=unit,status=record["status"],fit_calls=execution["classifier_fit_calls"])),flush=True)
    np.savez_compressed(ROOT/"predictions.npz", query_id=data["query_id"], unit_index=data["unit_index"],
        origin_index=data["origin_index"], split=data["split"], clipped=data["clipped"], original_q=data["original_q"],
        action_effect=data["action_effect"], original_pre=data["original_pre"], context_rows=data["context_rows"],
        native_high=data["native_high"], structured=data["structured"], probability=probability,
        prediction=prediction, fit_sample_weight=sample_weights)
    return probability, prediction, all_records


def probability_auc(y, probability):
    return float(roc_auc_score(y,probability)) if len(y) and np.any(y) and np.any(~y) and np.isfinite(probability).all() else None


def evaluate(data, rows, probability, prediction):
    results = []; table = []; disagreement = []
    for ui,row in enumerate(rows):
        record = dict(unit=row["unit"], assignment=row["assignment"])
        for scope in SCOPES:
            split = "assessment" if "assessment" in scope else "fit"
            cohort = (data["split"] == split) & (data["unit_index"] == ui)
            if scope.startswith("primary"): cohort &= data["clipped"]
            y = data["native_high"][cohort].reshape(-1)
            p = prediction[cohort].reshape(-1); s = data["structured"][cohort].reshape(-1)
            m = metrics(y,p); st = metrics(y,s)
            auroc = probability_auc(y,probability[cohort].reshape(-1))
            delta = m["balanced_accuracy"]-st["balanced_accuracy"] if m["balanced_accuracy"] is not None and st["balanced_accuracy"] is not None else None
            record[scope] = dict(paired_origins=int(cohort.sum()),clipped_paired_origins=int((cohort & data["clipped"]).sum()),
                                 continuous=m, structured=st, continuous_probability_auroc=auroc, delta=delta)
            table.append(dict(unit=row["unit"], assignment=row["assignment"], scope=scope,
                paired_origins=int(cohort.sum()), clipped_paired_origins=int((cohort & data["clipped"]).sum()),
                **m, structured_balanced_accuracy=st["balanced_accuracy"], delta=delta, probability_auroc=auroc))
            for label,qmask in [("one",data["original_q"] == 1),("many",data["original_q"] >= 2)]:
                selected = cohort & qmask
                for action,name in enumerate(["repair","neutral"]):
                    truth = data["native_high"][selected,action]
                    continuous = prediction[selected,action]; structured = data["structured"][selected,action].astype(bool)
                    item = dict(unit=row["unit"],assignment=row["assignment"],scope=scope,qclass=label,action=name,
                                n=len(truth),native_positive=int(truth.sum()),native_negative=int((~truth).sum()))
                    if np.isfinite(continuous).all():
                        continuous = continuous.astype(bool)
                        ce = continuous != truth; se = structured != truth
                        item.update(prediction_disagreement=int((continuous != structured).sum()),
                            continuous_errors=int(ce.sum()),structured_errors=int(se.sum()),
                            both_correct=int(((~ce)&(~se)).sum()), continuous_only_correct=int(((~ce)&se).sum()),
                            structured_only_correct=int((ce&(~se)).sum()),both_wrong=int((ce&se).sum()))
                    else:
                        item.update({k:None for k in ["prediction_disagreement","continuous_errors","structured_errors",
                            "both_correct","continuous_only_correct","structured_only_correct","both_wrong"]})
                    disagreement.append(item)
        results.append(record)
    aggregate = {}
    for scope in SCOPES:
        aggregate[scope] = {name:assignment_macro([r[scope][name]["balanced_accuracy"] for r in results],rows)
                            for name in ["continuous","structured"]}
        aggregate[scope]["delta"] = assignment_macro([r[scope]["delta"] for r in results],rows)
        aggregate[scope]["continuous_probability_auroc"] = assignment_macro([r[scope]["continuous_probability_auroc"] for r in results],rows)
    csv_write("unit_metrics.csv",table)
    csv_write("disagreement_by_unit.csv",disagreement)
    pooled = []
    counts = ["n","native_positive","native_negative","prediction_disagreement","continuous_errors",
              "structured_errors","both_correct","continuous_only_correct","structured_only_correct","both_wrong"]
    for scope in SCOPES:
        for qclass in ["one","many"]:
            for action in ["repair","neutral"]:
                selected = [r for r in disagreement if r["scope"] == scope and r["qclass"] == qclass and r["action"] == action]
                pooled.append(dict(scope=scope,qclass=qclass,action=action,
                    **{key:sum(r[key] for r in selected) if all(r[key] is not None for r in selected) else None for key in counts}))
    csv_write("disagreement_pooled.csv",pooled)
    return dict(units=results,aggregate=aggregate,pooled_disagreement=pooled,
                pooled_disagreement_warning="Pooled raw counts, not assignment-weighted accuracy")


def paired_bootstrap(data, rows, prediction):
    saved = dict(np.load(MECH/"bootstrap_raw.npz", allow_pickle=False))
    origin_ids = saved["origin_ids"]; weights = saved["origin_multiplicities"]
    assert weights.shape == (2000,len(origin_ids))
    assert np.issubdtype(weights.dtype,np.integer) and (weights >= 0).all()
    assert np.all(weights.sum(axis=1) == len(origin_ids))
    assert np.array_equal(origin_ids,np.unique(data["origin_index"][data["split"] == "assessment"]))
    lookup = {int(v):i for i,v in enumerate(origin_ids)}
    cm = np.zeros((len(origin_ids),17,2,4),dtype=np.int64)
    valid = np.ones((17,2),dtype=bool)
    pred = np.stack([prediction,data["structured"]],axis=-1)
    for k in np.flatnonzero((data["split"] == "assessment") & data["clipped"]):
        oi = lookup[int(data["origin_index"][k])]; ui = data["unit_index"][k]
        truth = data["native_high"][k]
        for mi in range(2):
            p = pred[k,:,mi]
            if not np.isfinite(p).all(): valid[ui,mi] = False; continue
            p = p.astype(bool)
            cm[oi,ui,mi] += [(truth&p).sum(),((~truth)&(~p)).sum(),((~truth)&p).sum(),(truth&(~p)).sum()]
    totals = (weights.astype(np.float64) @ cm.reshape(len(origin_ids),-1).astype(np.float64)).reshape(2000,17,2,4)
    positive = totals[:,:,:,0]+totals[:,:,:,3]; negative = totals[:,:,:,1]+totals[:,:,:,2]
    with np.errstate(invalid="ignore",divide="ignore"):
        ba = (totals[:,:,:,0]/positive + totals[:,:,:,1]/negative)/2
    ba[:,~valid] = np.nan
    assignments = sorted({r["assignment"] for r in rows})
    assignment_ba = np.stack([ba[:,[i for i,r in enumerate(rows) if r["assignment"] == a]].mean(axis=1)
                              for a in assignments],axis=1)
    macro = assignment_ba.mean(axis=1)
    delta = macro[:,0]-macro[:,1]
    original_structured_index = list(saved["model_names"]).index("structured")
    assert np.array_equal(ba[:,:,1],saved["per_unit_balanced_accuracy"][:,:,original_structured_index],equal_nan=True)
    assert np.array_equal(macro[:,1],saved["macro_balanced_accuracy"][:,original_structured_index],equal_nan=True)
    assert np.array_equal(np.asarray(assignments),saved["assignments"])
    np.savez_compressed(ROOT/"paired_bootstrap.npz",origin_ids=origin_ids,origin_multiplicities=weights,
        per_origin_confusion=cm,per_unit_balanced_accuracy=ba,assignment_balanced_accuracy=assignment_ba,
        macro_balanced_accuracy=macro,delta=delta,assignments=np.asarray(assignments),
        model_names=np.array(["continuous","structured"]))
    return dict(origin_multiplicities_reused_exactly=True, structured_bootstrap_matches_exactly=True,
        original_bootstrap_sha256=sha(MECH/"bootstrap_raw.npz"),distinct_origin_clusters=len(origin_ids),
        continuous=interval(macro[:,0]),structured=interval(macro[:,1]),delta=interval(delta),
        per_unit={str(r["unit"]):dict(continuous=interval(ba[:,i,0]),structured=interval(ba[:,i,1]),
                                    delta=interval(ba[:,i,0]-ba[:,i,1])) for i,r in enumerate(rows)})


def costs(data, fits, execution, start):
    stage_a = read_json(MECH/"stage_a.json")
    previous = read_json(MECH/"cost_ledger.json")
    supervised = []
    for ui,fit in enumerate(fits):
        old = stage_a["units"][ui]
        assert old["unit"] == fit["unit"]
        supervised.append(dict(unit=fit["unit"],assignment=fit["assignment"],
            readiness_original_fit_origins=old["fit"]["clipped_count"],
            readiness_q_one_labels=old["fit"]["q_one"],readiness_q_many_labels=old["fit"]["q_many"],
            continuous_edited_fit_action_rows=fit["fit_action_rows"],continuous_native_positive_labels=fit["positive"],
            continuous_native_negative_labels=fit["negative"],fit_seconds=fit["fit_seconds"],prediction_seconds=fit["prediction_seconds"]))
    csv_write("supervision.csv",supervised)
    serialized = [r["serialization"] for r in fits if "serialization" in r]
    native_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(native_peak if sys.platform == "darwin" else native_peak*1024)
    decoder = {key:sum(r[key] for r in serialized) for key in ["tree_count","leaf_count","internal_split_count",
        "raw_prediction_arrays_bytes","full_estimator_pickle_bytes","prediction_arrays_npz_bytes"]}
    assert decoder["tree_count"] <= 3400 and decoder["leaf_count"] <= 23800 and decoder["internal_split_count"] <= 20400
    decoder["maximum_tree_depth"] = max([r["maximum_depth"] for r in serialized],default=0)
    decoder["model_directory_bytes"] = sum(p.stat().st_size for p in (ROOT/"models").iterdir() if p.is_file())
    ledger = dict(neural_forwards_this_step=0,new_formulas=0,protected_cases_accessed=0,
        classifier_fit_calls=execution["classifier_fit_calls"],fit_slot_budget=17,assessment_passes=1,
        thread_limit=1,known_unit_report_bits=dict(readiness=1,continuous_float32=32),
        seventeen_unit_report_bits=dict(readiness=17,continuous_float32=544),
        with_original_34_bit_report=dict(readiness_state=51,continuous_state=578,
            readiness_with_fixed_action=67,continuous_with_fixed_action=594),
        action_fixed_bits=16,action_packed_bits=14,variable_target_extra_bits=5,
        static_readiness_float32_threshold_bits=544,static_readiness_rule_tag_bits=34,
        source_unit_assignment_table_bits=238,clause_truth_lookup_bits=1280,
        continuous_decoder=decoder,tree_comparisons_per_known_unit_upper=600,
        continuous_extra_computation="action encoding and assignment lookup, tree aggregation, sigmoid, strict probability comparison",
        shared_extraction="Both reports require original native checkpoint computation; no new extraction in this cached step",
        original_checkpoint_parameters=previous["original_model_parameters"],original_checkpoint_bytes=previous["checkpoint_bytes"],
        original_mechanism_native_forwards=previous["total_native_forwards"],
        supervision=supervised,readiness_total_unit_origin_q_labels=sum(r["readiness_original_fit_origins"] for r in supervised),
        continuous_total_unit_action_native_labels=sum(r["continuous_edited_fit_action_rows"] for r in supervised),
        supervision_count_warning="Summed labels count unit-origin or unit-action observations; they are not distinct input formulas or independent samples",
        fit_seconds=sum(r["fit_seconds"] for r in fits),prediction_seconds=sum(r["prediction_seconds"] for r in fits),
        elapsed_through_cost_ledger_seconds=time.monotonic()-start,peak_memory_bytes=peak_bytes,
        peak_memory_definition="Process-wide ru_maxrss including interpreter and libraries; native macOS units are bytes",
        archive_bytes_location="Final MANIFEST.json records exact archived file bytes excluding itself and the still-open transcript",
        semantic_formula_comparators=dict(ordered_fixed_bits=60,ordered_packed_bits=54,multiset_packed_bits=33,
            limits="Requires decoding and SAT computation; multiset does not preserve sequence-sensitive native responses"),
        capacity_matched=False,asymmetric_supervision=True)
    write_json(ROOT/"cost_ledger.json",ledger)
    return ledger


def report(result, bootstrap, interpretation_result, ledger, fits):
    primary = result["aggregate"]["primary_assessment"]
    ci = bootstrap["delta"]["interval_2_5_97_5"]
    text = "# Cached continuous-score comparison: development result\n\n"
    text += f"Continuous predictor balanced accuracy: **{primary['continuous']['macro']}**; unchanged readiness rule: **{primary['structured']['macro']}**. Delta (continuous minus readiness): **{primary['delta']['macro']}**, with descriptive 95% paired origin-cluster interval **{ci}**.\n\n"
    text += f"Contract interpretation: **{interpretation_result['category']}**. Readiness meets its separate two-point development noninferiority criterion: **{interpretation_result['readiness_meets_development_noninferiority']}**. Descriptive parity at two points: **{interpretation_result['descriptive_parity']}**.\n\n"
    explanations = {
        "ONE_BIT_LOSES_MATERIAL_ACCURACY":"The one-bit forecast loses material native predictive accuracy relative to this fitted comparator on this cohort. This contradicts two-point predictive sufficiency here, without refuting the approximate assignment interpretation.",
        "CONTINUOUS_MEETS_DEVELOPMENT_NONINFERIORITY":"This comparator meets the declared development noninferiority criterion. Generic continuous-margin prediction remains a viable account. That criterion alone does not establish that readiness is equally accurate.",
        "THIS_COMPARATOR_MISSES_DECLARED_PARITY":"This particular fitted comparator misses the declared parity target. Limited edited-fit supervision, capacity and optimization remain alternatives; unrestricted continuous-margin sufficiency is not falsified.",
        "UNRESOLVED_OR_UNSUPPORTED":"The comparison is unresolved at the declared resolution or lacks required support. No additional fits, cases or tuning follow."}
    text += explanations[interpretation_result["category"]]+"\n\n"
    text += "| Scope | Continuous BA | Structured BA | Delta | Continuous probability AUROC |\n|---|---:|---:|---:|---:|\n"
    for scope in SCOPES:
        a = result["aggregate"][scope]
        text += f"| {scope} | {a['continuous']['macro']} | {a['structured']['macro']} | {a['delta']['macro']} | {a['continuous_probability_auroc']['macro']} |\n"
    support = result["units"]
    text += "\nPrimary assessment retains "+str(sum(r["primary_assessment"]["paired_origins"] for r in support))+" clipped unit-origin pairs out of "+str(sum(r["all_selected_assessment"]["paired_origins"] for r in support))+" selected pairs. Both actions and repeated native input rows remain present. The 17 source units represent 16 assignments, with the two TTTFF units averaged first.\n\n"
    text += f"All {bootstrap['delta']['draws']} saved bootstrap vectors were reused over {bootstrap['distinct_origin_clusters']} origin IDs. Valid delta draws: {bootstrap['delta']['valid_draws']}; NA draws: {bootstrap['delta']['na_draws']}. The unchanged structured bootstrap matches exactly. These intervals condition on the fitted predictors and opened cohort; they omit fitting, selection and hypothesis-development uncertainty.\n\n"
    text += "| Unit | Original-fit q labels for readiness | Edited-fit native labels for continuous | Fit status |\n|---|---:|---:|---|\n"
    for r,f in zip(ledger["supervision"],fits):
        text += f"| {r['unit']} | {r['readiness_original_fit_origins']} | {r['continuous_edited_fit_action_rows']} | {f['status']} |\n"
    text += f"\nReadiness uses {ledger['readiness_total_unit_origin_q_labels']} unit-origin semantic q labels; continuous uses {ledger['continuous_total_unit_action_native_labels']} selected unit-action native labels. These are different labels and sample counts, not independent formula counts. A readiness advantage could reflect this asymmetry or finite-sample fitting.\n\n"
    d = ledger["continuous_decoder"]
    text += f"Exactly {ledger['classifier_fit_calls']} classifier fits; zero neural forwards and zero new formulas. Fitting took {ledger['fit_seconds']:.6f} seconds; probability prediction took {ledger['prediction_seconds']:.6f} seconds. Process peak memory: {ledger['peak_memory_bytes']} bytes. Actual ensembles: {d['tree_count']} trees, {d['leaf_count']} leaves, {d['internal_split_count']} internal split nodes. Full estimator pickles total {d['full_estimator_pickle_bytes']} bytes; prediction-array exports total {d['prediction_arrays_npz_bytes']} compressed bytes ({d['raw_prediction_arrays_bytes']} raw array bytes).\n\n"
    text += "Readiness charges 17 state bits versus 544 float32 score bits across the nominated units. Keeping the original report gives 51 versus 578 state bits, or 67 versus 594 with the same fixed action descriptor. Static thresholds, assignment lookup, clause lookup, all ensemble parameters, label supervision and common native extraction are charged separately. This is not a capacity-matched comparison or a global information minimum.\n\n"
    warning_count = sum(len(f["warnings"]) for f in fits)
    failures = [dict(unit=f["unit"],status=f["status"]) for f in fits if f["status"] != "FITTED_AND_PREDICTED"]
    text += f"Fit/prediction warning count: {warning_count}; nonstandard fit statuses: {failures}. Raw probabilities, strict >0.5 decisions, fit weights, complete model files, per-unit confusions, q/action disagreement tables and all paired draws are retained.\n\n"
    text += "This post-aggregate development comparison is not blind or protected confirmation. It establishes neither a unique latent mechanism nor global SAT repair, and does not address context invariance, all 34 published units or the earlier collision. The source's weighted-clause-count/threshold account remains the relevant prior art. The earlier post-ReLU magnitude pilot stays unchanged. The original pilot and mechanism manifests were verified before and after this run. Stop: no further learner, neural inference, context permutation or protected data is authorized by this result.\n"
    (ROOT/"REPORT.md").write_text(text)


def main():
    start = time.monotonic()
    with (ROOT/"run.lock").open("x") as lock:
        lock.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+"\n")
    execution = dict(status="RUNNING",classifier_fit_calls=0,neural_forwards=0,new_formulas=0,
        protected_cases_accessed=0,assessment_passes=0,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        freeze_sha256=sha(ROOT/"freeze.json"))
    write_json(ROOT/"execution.json",execution)
    try:
        verify_freeze(); before = verify_originals()
        assert "torch" not in sys.modules
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            assert all(p["num_threads"] == 1 for p in pools)
            write_json(ROOT/"runtime.json",dict(versions=versions(),threadpools=pools,prior_manifests_before=before))
            data,rows = cached_data()
            probability,prediction,fits = fit_once(data,rows,execution)
            execution["assessment_passes"] = 1
            write_json(ROOT/"execution.json",execution)
            result = evaluate(data,rows,probability,prediction)
            bootstrap = paired_bootstrap(data,rows,prediction)
            primary = result["aggregate"]["primary_assessment"]
            judgment = interpretation(primary["delta"]["macro"],bootstrap["delta"]["interval_2_5_97_5"])
            write_json(ROOT/"results.json",dict(result,bootstrap=bootstrap,interpretation=judgment))
            ledger = costs(data,fits,execution,start)
            verify_freeze(); after = verify_originals(); assert before == after
            report(result,bootstrap,judgment,ledger,fits)
        execution.update(status="STOPPED_AFTER_ONE_CACHED_COMPARISON",prior_manifests_after=after,
            elapsed_seconds=time.monotonic()-start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(ROOT/"execution.json",execution)
        all_files = [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and p.name not in ["MANIFEST.json","run.log"]]
        # run.log is still open while this process runs; its transcript is not in the immutable scientific manifest.
        manifest = dict(status="STOPPED_AFTER_ONE_CACHED_COMPARISON",freeze_sha256=sha(ROOT/"freeze.json"),
            files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(all_files)},
            excluded_live_transcript="run.log",archive_bytes_excluding_manifest_and_transcript=sum(p.stat().st_size for p in all_files))
        write_json(ROOT/"MANIFEST.json",manifest)
        print(json.dumps(dict(status=execution["status"],primary=primary,bootstrap_delta=bootstrap["delta"],
            interpretation=judgment,manifest_sha256=sha(ROOT/"MANIFEST.json"))),flush=True)
    except Exception as error:
        execution.update(status="ERROR_STOP",exception=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        write_json(ROOT/"execution.json",execution)
        raise


if __name__ == "__main__": main()
