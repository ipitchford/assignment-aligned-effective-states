"""Independent cached-output arithmetic audit. No study imports, fit, or inference."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "sat_margin_baseline"
OLD = HERE.parent / "sat_mechanism"
EXPECTED = "3a2c115056d9512d72095f96e8a6766691663122a2c10997496e70e21a31edc5"
checked = []


def check(condition, name):
    if not bool(condition):
        raise AssertionError(name)
    checked.append(name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    return np.allclose(a, b, rtol=0, atol=3e-15, equal_nan=True)


def confusion(y, p):
    y = np.asarray(y, bool)
    p = np.asarray(p, bool)
    tp, tn = int(np.sum(y & p)), int(np.sum(~y & ~p))
    pos, neg = int(y.sum()), int((~y).sum())
    return dict(n=len(y), positive=pos, negative=neg, tp=tp, tn=tn,
                fp=neg-tn, fn=pos-tp, sensitivity=tp/pos if pos else None,
                specificity=tn/neg if neg else None,
                balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None)


def auc_pairwise(y, p):
    positive = np.asarray(p)[np.asarray(y, bool)]
    negative = np.asarray(p)[~np.asarray(y, bool)]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean((positive[:, None] > negative) + .5*(positive[:, None] == negative)))


PRESENTATION_PATH = HERE.parent / 'PUBLIC_PRESENTATION_DERIVATIVES.json'
if sha(PRESENTATION_PATH) != '324de49b83ab2bc5d63606852d56f837ab6c91ccca031defc1e83a2798eef618':
    raise RuntimeError('Presentation map identity mismatch')
PRESENTATION = read(PRESENTATION_PATH)

def historical_identity(path):
    """Return recorded identity only for an explicitly pinned presentation derivative."""
    path=Path(path)
    actual=sha(path)
    for row in PRESENTATION['files']:
        if actual == row['public_sha256']:
            if path.stat().st_size != row['public_bytes']:
                raise RuntimeError('Presentation derivative size mismatch')
            return row['original_sha256'], row['original_bytes']
    return actual, path.stat().st_size

manifest = read(BASE / "MANIFEST.json")
check(sha(BASE / "MANIFEST.json") == EXPECTED, "bound baseline manifest")
for name, entry in manifest["files"].items():
    if name.endswith(".pkl"):
        continue  # Historical byte identity unavailable; no unpickling required.
    p = BASE/name
    check(historical_identity(p) == (entry["sha256"], entry["bytes"]), "manifest:"+name)
check(sum(x["bytes"] for x in manifest["files"].values()) == manifest["archive_bytes_excluding_manifest_and_transcript"], "archive byte total")
freeze = read(BASE/"freeze.json")
execution = read(BASE/"execution.json")
check(sha(BASE/"freeze.json") == execution["freeze_sha256"] == manifest["freeze_sha256"], "freeze bound to execution")
for kind in ["local_sha256", "input_sha256"]:
    for name, expected in freeze[kind].items():
        check(historical_identity(BASE/name)[0] == expected, kind+":"+name)
check(freeze["created_utc"] < execution["start_utc"] < execution["end_utc"], "recorded chronology")
check(execution["classifier_fit_calls"] == 17 and execution["assessment_passes"] == 1, "recorded bounded fits")
check(all(execution[k] == 0 for k in ["neural_forwards", "new_formulas", "protected_cases_accessed"]), "recorded zero new neural data")

data = dict(np.load(BASE/"predictions.npz", allow_pickle=False))
original = dict(np.load(OLD/"stage_b_raw.npz", allow_pickle=False))
queries = read(OLD/"selected_queries.json")["rows"]
units = read(OLD/"source_mapping.json")["single_assignment_general_units"]
fits = read(BASE/"fit_records.json")
stage_a = read(OLD/"stage_a.json")["units"]
results = read(BASE/"results.json")
ledger = read(BASE/"cost_ledger.json")
unit_ids = [u["unit"] for u in units]
assignments = sorted({u["assignment"] for u in units})
check(len(queries) == 2176 and len(units) == 17 and len(assignments) == 16, "cohort dimensions")
check([u["unit"] for u in units if u["assignment"] == "TTTFF"] == [55, 416], "duplicate assignment retained")
for k in ["query_id", "unit_index", "origin_index", "split", "clipped", "original_q", "action_effect", "original_pre", "context_rows", "native_high"]:
    check(np.array_equal(data[k], original[k]), "unchanged cached:"+k)
check(np.array_equal(data["query_id"], np.arange(2176)), "ascending query order")
check(np.array_equal(data["split"] == "fit", data["origin_index"] < 3072), "unchanged origin split")
check(not set(data["origin_index"][data["split"] == "fit"]) & set(data["origin_index"][data["split"] == "assessment"]), "disjoint fit assessment origins")
check(not {q["origin_hash"] for q in queries if q["split"] == "fit"} & {q["origin_hash"] for q in queries if q["split"] == "assessment"}, "disjoint origin content hashes")
old_manifest = read(OLD/"MANIFEST.json")
check(sha(OLD/"query_tokens.npz") == old_manifest["files"]["query_tokens.npz"]["sha256"], "bound cached native input mapping")
query_tokens = dict(np.load(OLD/"query_tokens.npz", allow_pickle=False))
unique_id = query_tokens["query_action_to_unique"]
check(np.array_equal(query_tokens["unique_tokens"][unique_id.reshape(-1)], query_tokens["all_tokens"]), "native input mapping identity")
check(not set(unique_id[data["split"] == "fit"].reshape(-1)) & set(unique_id[data["split"] == "assessment"].reshape(-1)), "no identical selected edited input across splits")
check(np.array_equal(data["clipped"], (data["original_q"] >= 1) & (data["original_pre"] < 0)), "primary cohort fixed by original state")
check(np.array_equal(data["native_high"], original["native_edited_post"] >= .5), "same native response target")
si = list(original["model_names"]).index("structured")
check(np.array_equal(data["structured"], original["predictions"][:, :, si]), "structured forecast unchanged")
check(data["probability"].dtype == np.float64 and np.isfinite(data["probability"]).all(), "finite float64 probabilities")
check(np.all((data["probability"] >= 0) & (data["probability"] <= 1)), "probability range")
check(np.array_equal(data["prediction"], data["probability"] > .5), "strict probability decision including ties")

# Rebuild allowed columns only, from original float32 score and explicit action ledger.
clauses = [(i+5*a, j+5*b) for i in range(5) for j in range(i+1, 5) for a in range(2) for b in range(2)]
x = np.zeros((4352, 92), dtype=np.float64)
expected_effect = np.zeros((2176, 2), dtype=int)
expected_context = np.zeros((2176, 2, 3), dtype=int)
for k, q in enumerate(queries):
    check(all(data[field][k] == q[field] for field in ["query_id", "unit_index", "origin_index", "split", "original_q"]), f"ledger alignment:{k}")
    assignment = q["assignment"]
    check(assignment == units[q["unit_index"]]["assignment"] and q["unit"] == unit_ids[q["unit_index"]], f"source unit:{k}")
    def clause_value(cid):
        return any((assignment[l % 5] == "T") == (l < 5) for l in clauses[cid])
    for a, new in enumerate([q["repair_clause"], q["neutral_clause"]]):
        expected_effect[k, a] = int(not clause_value(q["old_clause"])) - int(not clause_value(new))
        expected_context[k, a] = [q["slot"], q["old_clause"], new]
        j = 2*k+a
        x[j, 0] = float(data["original_pre"][k])
        x[j, 1] = expected_effect[k, a]
        x[j, 2+q["slot"]] = x[j, 12+q["old_clause"]] = x[j, 52+new] = 1
check(np.array_equal(expected_effect, data["action_effect"]), "action effect from assignment and old/new clause only")
check(np.array_equal(expected_context, data["context_rows"]), "repair-neutral context order")
check(np.array_equal(x[:, 0].astype(np.float32), np.repeat(data["original_pre"], 2)), "exact float32 score promotion")
check(np.array_equal(x[:, 1].reshape(-1, 2), np.tile([1, 0], (2176, 1))), "action feature identity")
check(np.isfinite(x).all() and all(np.all(x[:, lo:hi].sum(1) == 1) for lo, hi in [(2, 12), (12, 52), (52, 92)]), "92 allowed finite features and one-hot columns")

model_counts = dict(tree_count=0, leaf_count=0, internal_split_count=0, raw_prediction_arrays_bytes=0, full_estimator_pickle_bytes=0, prediction_arrays_npz_bytes=0)
for ui, u in enumerate(units):
    unit = u["unit"]
    all_rows = data["unit_index"] == ui
    fit_rows = all_rows & (data["split"] == "fit") & data["clipped"]
    y = data["native_high"][fit_rows]
    pos, neg, n = int(y.sum()), int((~y).sum()), y.size
    f = fits[ui]
    check(f["unit"] == unit and f["status"] == "FITTED_AND_PREDICTED" and f["fit_attempts"] == 1, f"fit status:{unit}")
    check(f["parameters"] == freeze["estimator_parameters"], f"fixed learner:{unit}")
    check([f["fit_action_rows"], f["positive"], f["negative"], f["fit_paired_origins"]] == [n, pos, neg, n//2], f"fit supervision:{unit}")
    weights = np.where(y, n/(2*pos), n/(2*neg))
    check(np.array_equal(data["fit_sample_weight"][fit_rows], weights), f"fit-only class weights:{unit}")
    check(np.isnan(data["fit_sample_weight"][all_rows & ~fit_rows]).all(), f"no assessment weights:{unit}")
    raw = np.load(BASE/f"models/unit_{unit:03d}_prediction_raw.npz", allow_pickle=False)
    check(np.array_equal(raw["query_id"], data["query_id"][all_rows]), f"raw probability query order:{unit}")
    check(np.array_equal(raw["classes"], [False, True]) and raw["probabilities"].shape == (int(all_rows.sum())*2, 2), f"raw probability classes:{unit}")
    check(np.array_equal(raw["probabilities"][:, 1].reshape(-1, 2), data["probability"][all_rows]), f"raw probability assembly:{unit}")
    check(close(raw["probabilities"].sum(axis=1), 1), f"probability complements:{unit}")
    check(f["probability_point5_ties"] == int((raw["probabilities"][:, 1] == .5).sum()), f"tie record:{unit}")
    m = read(BASE/f"models/unit_{unit:03d}.json")
    arrays = dict(np.load(BASE/f"models/unit_{unit:03d}.npz", allow_pickle=False))
    fit_x = x[np.repeat(fit_rows, 2)]
    bin_matches = []
    for column in range(92):
        distinct = np.unique(fit_x[:, column])
        # At most 128 rows < max_bins=255: every boundary is a midpoint.
        expected_bins = (distinct[:-1] + distinct[1:])*.5
        bin_matches.append(np.array_equal(arrays[f"bin_thresholds_{column:03d}"], expected_bins))
    check(all(bin_matches), f"all bin thresholds derive from fit features only:{unit}")
    trees = [v for key, v in arrays.items() if key.endswith("_nodes")]
    check(m["parameters"] == freeze["estimator_parameters"] and m["n_features_in"] == 92 and len(trees) == 200, f"model metadata:{unit}")
    check(all(int(t["count"][0]) == n and int(t["depth"].max()) <= 3 and int(t["is_leaf"].sum()) <= 7 for t in trees), f"tree fit sample and capacity:{unit}")
    check(all(np.all(t["feature_idx"][t["is_leaf"] == 0] < 92) and np.all(t["feature_idx"][t["is_leaf"] == 0] >= 0) and not np.any(t["is_categorical"]) for t in trees), f"tree feature bounds:{unit}")
    counts = dict(tree_count=len(trees), leaf_count=sum(int(t["is_leaf"].sum()) for t in trees), internal_split_count=sum(int((t["is_leaf"] == 0).sum()) for t in trees), raw_prediction_arrays_bytes=sum(v.nbytes for v in arrays.values()), full_estimator_pickle_bytes=manifest["files"][f"models/unit_{unit:03d}.pkl"]["bytes"], prediction_arrays_npz_bytes=(BASE/f"models/unit_{unit:03d}.npz").stat().st_size)
    check(all(counts[k] == m[k] == f["serialization"][k] for k in counts), f"actual decoder sizes:{unit}")
    for k, v in counts.items():
        model_counts[k] += v
    cost = ledger["supervision"][ui]
    old = stage_a[ui]["fit"]
    check([cost["unit"], cost["readiness_original_fit_origins"], cost["readiness_q_one_labels"], cost["readiness_q_many_labels"], cost["continuous_edited_fit_action_rows"]] == [unit, old["clipped_count"], old["q_one"], old["q_many"], n], f"supervision ledger:{unit}")

recomputed = []
scopes = ["primary_assessment", "all_selected_assessment", "primary_fit", "all_selected_fit"]
disagreements = list(csv.DictReader((BASE/"disagreement_by_unit.csv").open()))
metric_csv = list(csv.DictReader((BASE/"unit_metrics.csv").open()))
aggregate = {}
for scope in scopes:
    split = "assessment" if "assessment" in scope else "fit"
    cohort = (data["split"] == split) & (data["clipped"] if scope.startswith("primary") else True)
    scope_rows = []
    for ui, u in enumerate(units):
        take = cohort & (data["unit_index"] == ui)
        y = data["native_high"][take].reshape(-1)
        p = data["prediction"][take].reshape(-1)
        s = data["structured"][take].reshape(-1)
        cm, sm = confusion(y, p), confusion(y, s)
        auroc = auc_pairwise(y, data["probability"][take].reshape(-1))
        saved = results["units"][ui][scope]
        check(all(close(cm[k], saved["continuous"][k]) for k in cm), f"continuous confusion:{u['unit']}:{scope}")
        check(all(close(sm[k], saved["structured"][k]) for k in sm), f"structured confusion:{u['unit']}:{scope}")
        check(close(auroc, saved["continuous_probability_auroc"]), f"pairwise AUROC:{u['unit']}:{scope}")
        row = dict(unit=u["unit"], assignment=u["assignment"], scope=scope, paired_origins=int(take.sum()), continuous_ba=cm["balanced_accuracy"], structured_ba=sm["balanced_accuracy"], delta=cm["balanced_accuracy"]-sm["balanced_accuracy"], auc=auroc)
        scope_rows.append(row); recomputed.append(row)
        csv_row = next(r for r in metric_csv if int(r["unit"]) == u["unit"] and r["scope"] == scope)
        check(all(close(float(csv_row[k]), cm[k]) for k in cm) and close(float(csv_row["delta"]), row["delta"]), f"unit CSV:{u['unit']}:{scope}")
        for qclass in ["one", "many"]:
            qtake = take & ((data["original_q"] == 1) if qclass == "one" else (data["original_q"] >= 2))
            for ai, action in enumerate(["repair", "neutral"]):
                truth = data["native_high"][qtake, ai]
                ce = data["prediction"][qtake, ai] != truth
                se = data["structured"][qtake, ai] != truth
                want = dict(n=len(truth), native_positive=int(truth.sum()), native_negative=int((~truth).sum()), prediction_disagreement=int((ce != se).sum()), continuous_errors=int(ce.sum()), structured_errors=int(se.sum()), both_correct=int((~ce & ~se).sum()), continuous_only_correct=int((~ce & se).sum()), structured_only_correct=int((ce & ~se).sum()), both_wrong=int((ce & se).sum()))
                got = next(r for r in disagreements if int(r["unit"]) == u["unit"] and r["scope"] == scope and r["qclass"] == qclass and r["action"] == action)
                check(all(int(got[k]) == v for k, v in want.items()), f"disagreement:{u['unit']}:{scope}:{qclass}:{action}")
    aggregate[scope] = {}
    for field, saved_field in [("continuous_ba", "continuous"), ("structured_ba", "structured"), ("delta", "delta"), ("auc", "continuous_probability_auroc")]:
        per_assignment = {a: float(np.mean([r[field] for r in scope_rows if r["assignment"] == a])) for a in assignments}
        macro = float(np.mean(list(per_assignment.values())))
        got = results["aggregate"][scope][saved_field]
        check(all(close(per_assignment[a], got["assignments"][a]) for a in assignments) and close(macro, got["macro"]), "assignment macro:"+scope+":"+field)
        aggregate[scope][field] = macro

pooled_csv = list(csv.DictReader((BASE/"disagreement_pooled.csv").open()))
for row in pooled_csv:
    selected = [r for r in disagreements if all(r[k] == row[k] for k in ["scope", "qclass", "action"])]
    check(all(int(row[k]) == sum(int(r[k]) for r in selected) for k in row if k not in ["scope", "qclass", "action"]), "pooled disagreement:"+str([row[k] for k in ["scope", "qclass", "action"]]))

# Independent direct weighted-row bootstrap; no reuse of submitted per-origin confusions.
old_boot = dict(np.load(OLD/"bootstrap_raw.npz", allow_pickle=False))
boot = dict(np.load(BASE/"paired_bootstrap.npz", allow_pickle=False))
origins, multiplicities = old_boot["origin_ids"], old_boot["origin_multiplicities"]
check(np.array_equal(origins, boot["origin_ids"]) and np.array_equal(multiplicities, boot["origin_multiplicities"]), "exact original bootstrap draws")
check(multiplicities.shape == (2000, 685) and np.all(multiplicities >= 0) and np.all(multiplicities.sum(1) == 685), "origin multiplicities valid")
ba = np.empty((2000, 17, 2))
for ui in range(17):
    take = (data["unit_index"] == ui) & (data["split"] == "assessment") & data["clipped"]
    loc = np.searchsorted(origins, data["origin_index"][take])
    check(np.array_equal(origins[loc], data["origin_index"][take]), f"bootstrap origin lookup:{ui}")
    w = np.repeat(multiplicities[:, loc].astype(np.int64), 2, axis=1)
    truth = data["native_high"][take].reshape(-1)
    positive, negative = w[:, truth].sum(1), w[:, ~truth].sum(1)
    for mi, field in enumerate(["prediction", "structured"]):
        pred = data[field][take].reshape(-1).astype(bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            ba[:, ui, mi] = .5*(w[:, truth & pred].sum(1)/positive + w[:, ~truth & ~pred].sum(1)/negative)
check(np.array_equal(ba, boot["per_unit_balanced_accuracy"], equal_nan=True), "all unit draw BAs exact")
assignment_ba = np.stack([np.mean(ba[:, [i for i, u in enumerate(units) if u["assignment"] == a], :], axis=1) for a in assignments], axis=1)
macro = assignment_ba.mean(1)
delta = macro[:, 0] - macro[:, 1]
check(np.array_equal(assignment_ba, boot["assignment_balanced_accuracy"], equal_nan=True), "duplicate-first bootstrap aggregation")
check(np.array_equal(macro, boot["macro_balanced_accuracy"], equal_nan=True), "all macro draws exact")
check(np.array_equal(delta, boot["delta"], equal_nan=True), "all paired differences exact")
old_index = list(old_boot["model_names"]).index("structured")
check(np.array_equal(macro[:, 1], old_boot["macro_balanced_accuracy"][:, old_index], equal_nan=True), "unchanged structured draws")
valid = np.isfinite(delta)
ci = np.quantile(delta[valid], [.025, .975]).tolist() if valid.sum() >= 1900 else None
check(int(valid.sum()) == results["bootstrap"]["delta"]["valid_draws"] == 2000 and close(ci, results["bootstrap"]["delta"]["interval_2_5_97_5"]), "interval and valid denominator")
judgment = "ONE_BIT_LOSES_MATERIAL_ACCURACY" if ci[0] > .02 else "CONTINUOUS_MEETS_DEVELOPMENT_NONINFERIORITY" if ci[0] >= -.02 else "THIS_COMPARATOR_MISSES_DECLARED_PARITY" if ci[1] < -.02 else "UNRESOLVED_OR_UNSUPPORTED"
check(judgment == results["interpretation"]["category"], "outcome-neutral contract classification")
check(results["interpretation"]["readiness_meets_development_noninferiority"] == (ci[1] <= .02) and results["interpretation"]["descriptive_parity"] == (ci[0] >= -.02 and ci[1] <= .02), "separate readiness criterion")
check(all(ledger["continuous_decoder"][k] == v for k, v in model_counts.items()), "decoder cost totals")
check(ledger["continuous_decoder"]["model_directory_bytes"] == sum(p.stat().st_size for p in (BASE/"models").iterdir() if p.is_file()) + sum(v["bytes"] for k,v in manifest["files"].items() if k.endswith(".pkl")), "model directory bytes")
check(ledger["readiness_total_unit_origin_q_labels"] == sum(u["fit"]["clipped_count"] for u in stage_a) == 45693, "readiness supervision total")
check(ledger["continuous_total_unit_action_native_labels"] == sum(f["fit_action_rows"] for f in fits) == 1922, "continuous supervision total")
check(close(ledger["fit_seconds"], sum(f["fit_seconds"] for f in fits)) and close(ledger["prediction_seconds"], sum(f["prediction_seconds"] for f in fits)), "recorded runtime sums")
check(math.ceil(math.log2(40**10)) == 54 and math.ceil(math.log2(math.comb(49, 10))) == 33 and math.ceil(math.log2(10*40*40)) == 14, "packed comparator sizes")
check(17*32 == 544 and 17*(9+5) == 238 and 40*32 == 1280 and 34+17+16 == 67 and 34+544+16 == 594, "state and shared lookup arithmetic")
check(sha(BASE/"MANIFEST.json") == EXPECTED, "baseline manifest unchanged after audit")

summary = dict(status="PASS", checks=len(checked), audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), manifest_sha256=EXPECTED, checker_sha256=sha(__file__), scope="Cached arithmetic, row alignment, source-code inspection and artifact counts; no fits or model inference", classifier_fits=0, neural_forwards=0, tree_predictions=0, original_sat_audit_repeated=False, aggregate=aggregate, bootstrap=dict(draws=2000, valid_draws=int(valid.sum()), delta_interval=ci, classification=judgment), model_counts=model_counts, primary_selected_pairs=1088, primary_clipped_pairs=int(((data["split"] == "assessment") & data["clipped"]).sum()), probability_half_ties=int((data["probability"] == .5).sum()), primary_unit_delta=dict(positive=sum(r["delta"] > 0 for r in recomputed if r["scope"] == "primary_assessment"), zero=sum(r["delta"] == 0 for r in recomputed if r["scope"] == "primary_assessment"), negative=sum(r["delta"] < 0 for r in recomputed if r["scope"] == "primary_assessment")), archive_bytes=manifest["archive_bytes_excluding_manifest_and_transcript"], assertions=checked)
(HERE/"checks.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
with (HERE/"unit_metrics_recomputed.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(recomputed[0])); writer.writeheader(); writer.writerows(recomputed)
np.savez_compressed(HERE/"bootstrap_recomputed.npz", per_unit_balanced_accuracy=ba, assignment_balanced_accuracy=assignment_ba, macro_balanced_accuracy=macro, delta=delta)
print(json.dumps({k: v for k, v in summary.items() if k not in ["assertions", "aggregate"]}, indent=2))
