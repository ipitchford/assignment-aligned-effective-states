"""Independent saved-array audit. No study code import, model or new formulas.

Portable adaptation of the archived independent checker. The Boolean bank is
checked separately by the unchanged pre-frozen verifier. All event, mapping,
task, saved-resampling and quantile comparisons are retained. The declared RNG
is NOT regenerated: only the saved bootstrap indices are read. No new draws.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import hashlib
import itertools
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
OUT = args.output_dir.resolve()
WORK = args.work_root.resolve()
if not OUT.is_dir() or not (OUT / "bank_replay.json").is_file():
    raise SystemExit("Run the portable replay wrapper into a fresh output directory first.")
STUDY = WORK / "sat_renaming_launch_recovery"
ORIGINAL = WORK / "sat_renaming"
CONTRACT_SHA = "4b9dc2df3002dbbe56cbcc5d38c119ace801f16e2f3bd018d345deb4eda7af8c"
UNITS = [261, 291, 411, 492]
MAPS = {"correct": [1, 0, 3, 2], "identity": [0, 1, 2, 3], "wrong": [3, 2, 1, 0]}
checks, inputs, differences = [], {}, {}
started = time.monotonic()



# Public presentation adapter: scientific bytes remain exact; old prose identity
# is resolved only through a pinned map, not claimed as a current byte hash.
def presentation_identity(data):
    actual = hashlib.sha256(data).hexdigest()
    for parent in Path(__file__).resolve().parents:
        mapping = parent / "PUBLIC_PRESENTATION_DERIVATIVES.json"
        if mapping.is_file():
            raw = mapping.read_bytes()
            if hashlib.sha256(raw).hexdigest() != "324de49b83ab2bc5d63606852d56f837ab6c91ccca031defc1e83a2798eef618":
                raise RuntimeError("Presentation map identity mismatch")
            for row in json.loads(raw)["files"]:
                if actual == row["public_sha256"] and len(data) == row["public_bytes"]:
                    return row["original_sha256"]
            return actual
    raise RuntimeError("Missing pinned presentation map")


def sha(path):
    path = Path(path).resolve()
    value = presentation_identity(path.read_bytes())
    inputs[str(path)] = {"sha256": value, "bytes": path.stat().st_size}
    return value


def readj(path):
    sha(path)
    return json.loads(Path(path).read_text())


def arrays(path):
    sha(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def check(name, condition, detail=None):
    checks.append({"name": name, "pass": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(name + ": " + str(detail))


def equal(name, actual, expected, tolerance=0):
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(name + ": shape", actual.shape == expected.shape, [list(actual.shape), list(expected.shape)])
    if actual.dtype.kind in "US" or expected.dtype.kind in "US":
        check(name, np.array_equal(actual, expected))
        return
    delta = float(np.max(np.abs(actual.astype(float) - expected.astype(float)))) if actual.size else 0.0
    differences[name] = delta
    check(name, delta <= tolerance, {"max_absolute_difference": delta, "tolerance": tolerance})


def writej(name, data):
    with (OUT / name).open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def possible_classes(logits):
    """Contract intervals, explicitly comparing each class with the other 15."""
    values = [float(x) for x in logits]
    low = [x - (0.0001 + 0.00001 * abs(x)) for x in values]
    high = [x + (0.0001 + 0.00001 * abs(x)) for x in values]
    return tuple(high[i] >= max(low[j] for j in range(16) if j != i) for i in range(16))


def task_vectors(logits, labels):
    pred = np.empty(labels.shape, dtype=np.int64)
    possible = np.empty(labels.shape + (16,), dtype=bool)
    correct = np.empty(labels.shape + (3,), dtype=np.int8)
    for index in np.ndindex(labels.shape):
        row = logits[index]
        p = max(range(16), key=lambda k: float(row[k]))
        options = possible_classes(row)
        true = int(labels[index])
        pred[index] = p
        possible[index] = options
        correct[index] = (p == true, sum(options) == 1 and options[true], options[true])
    return pred, possible, correct


def finite_native_and_replay(native, replay, selected):
    check("native units and order", native["units"].tolist() == UNITS and native["block_ids"].tolist() == list(range(64)))
    check("fixed replay blocks", replay["units"].tolist() == UNITS and replay["block_ids"].tolist() == [0, 1]
          and selected["replay_block_ids"] == [0, 1])
    dimensions = {"pre": 4, "post": 4, "full_post": 512, "resid_mid": 128, "logits": 16}
    replay_diffs = {}
    for key, width in dimensions.items():
        check("native shape/dtype/finite " + key, native[key].shape == (64, 2, 4, width)
              and native[key].dtype == np.float32 and np.isfinite(native[key]).all())
        check("replay shape/dtype/finite " + key, replay[key].shape == (2, 2, 4, width)
              and replay[key].dtype == np.float32 and np.isfinite(replay[key]).all())
        equal("saved independent-process replay " + key, replay[key], native[key][:2])
        replay_diffs[key] = differences["saved independent-process replay " + key]
    equal("exact selected ReLU", native["post"], np.maximum(native["pre"], 0))
    equal("selected/full-post identity", native["post"], native["full_post"][..., UNITS])
    equal("saved threshold distances", native["post_distance_to_half"], np.abs(native["post"].astype(float) - 0.5))
    sorted_logits = np.sort(native["logits"].astype(float), axis=-1)
    equal("saved winning gaps", native["winning_logit_gap"], sorted_logits[..., -1] - sorted_logits[..., -2])
    return replay_diffs


def provenance():
    check("exact contract", sha(WORK / "next_phase/sat_renaming_execution_contract.md") == CONTRACT_SHA)
    for name, directory in [("freeze.json", STUDY), ("selected_data_freeze.json", STUDY), ("MANIFEST.json", STUDY), ("MANIFEST.json", ORIGINAL)]:
        manifest = readj(directory / name)
        mismatches = []
        for record in manifest["files"]:
            p = directory / record["path"]
            if p.resolve() == (STUDY / "native_replay.log").resolve() and not p.exists():
                continue  # Historical diagnostic log unavailable, not numerical data.
            data = p.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            size = len(data)
            mapping = readj(WORK.parent.parent / "PUBLIC_PRESENTATION_DERIVATIVES.json")
            for row in mapping["files"]:
                if actual == row["public_sha256"]:
                    size = row["original_bytes"]
            operational = p.resolve() == (STUDY / "verify_bank.py").resolve()
            if operational:
                public_manifest = readj(WORK.parent / "PACKAGE_MANIFEST.json")
                rel = p.resolve().relative_to(WORK.parent).as_posix()
                bound = next(r for r in public_manifest["files"] if r["path"] == rel)
                valid = actual == bound["sha256"] and len(data) == bound["bytes"]
            else:
                valid = sha(p) == record["sha256"] and size == record["bytes"]
            if not valid: mismatches.append(record["path"])
        check("distributed bound files " + str(directory.name) + "/" + name, not mismatches, {"historical_files": len(manifest["files"]), "unavailable_log": "native_replay.log", "mismatches": mismatches})
    recovery = readj(STUDY / "recovery_freeze.json")
    check("recovery scientific copies historical-identity equivalent", all(sha(STUDY / r["path"]) == sha(ORIGINAL / r["path"]) == r["sha256"] for r in recovery["unchanged_copies"] if r["path"] != "verify_bank.py"))
    old = readj(ORIGINAL / "execution.json")
    check("failed launch had zero native charges", old["status"] == "UNRESOLVED_RUNTIME_FAILURE"
          and old["batch_attempts"] == old["completed_input_forwards"] == old["conservative_input_forward_charge"] == 0)
    oldlog = (ORIGINAL / "native_initial.log").read_text()
    check("original pre-main import failure retained", "ModuleNotFoundError: No module named 'torch'" in oldlog
          and not (ORIGINAL / "native_initial_started.json").exists())
    execution = readj(STUDY / "execution.json")
    check("recovery native deadline preserved", execution["original_native_deadline"] == recovery["original_native_deadline"]
          and datetime.fromisoformat(execution["utc"]) < datetime.fromisoformat(recovery["original_native_deadline"]))
    check("bounded execution counts", execution["completed_input_forwards"] == execution["conservative_input_forward_charge"] == 528
          and execution["batch_attempts"] == 66 and execution["learner_fits"] == execution["new_proposals"] == 0)
    totals = Counter()
    for stage, nblocks in [("initial", 64), ("replay", 2)]:
        path = STUDY / ("native_" + stage + "_events.jsonl")
        sha(path)
        events = [json.loads(line) for line in path.read_text().splitlines()]
        attempts = [e for e in events if e["event"] == "BATCH_ATTEMPT"]
        completed = [e for e in events if e["event"] == "BATCH_COMPLETE"]
        check("exact journal count " + stage, len(attempts) == len(completed) == nblocks)
        check("exact journal block order " + stage, [e["block_id"] for e in attempts] == [e["block_id"] for e in completed] == list(range(nblocks)))
        check("eight-row charges " + stage, all(e["conservative_input_forward_charge"] == 8 and e["batch_call_charge"] == 1 for e in attempts)
              and all(e["input_forwards"] == 8 and e["batch_calls"] == 1 for e in completed))
        totals["forwards"] += sum(e["input_forwards"] for e in completed)
        totals["batches"] += len(completed)
    return dict(totals)


def events_and_metrics(bank, native, saved_events):
    bits = np.empty((64, 2, 4, 4), dtype=np.int8)
    lower, upper = np.empty_like(bits), np.empty_like(bits)
    correctness = np.empty(bits.shape + (3,), dtype=np.int8)
    for ix in np.ndindex(bits.shape):
        value = float(native["post"][ix]); rad = 0.0001 + 0.00001 * abs(value)
        options = [b for b in (0, 1) if (b == 0 and value - rad < .5) or (b == 1 and value + rad >= .5)]
        nominal = int(value >= .5)
        bits[ix], lower[ix], upper[ix] = nominal, min(options), max(options)
        truth = int(bank["expected_unit_bits"][ix])
        correctness[ix] = (nominal == truth, all(b == truth for b in options), any(b == truth for b in options))
    pred, possible, task_good = task_vectors(native["logits"], bank["labels"])
    for name, actual in [("nominal_bits", bits), ("lower_bits", lower), ("upper_bits", upper), ("correctness", correctness),
                         ("nominal_task_class", pred), ("possible_task_classes", possible), ("task_correctness", task_good)]:
        equal("event array " + name, actual, saved_events[name])
    # On this bank the admissible sets are singletons, so every mapping,
    # pattern, signed-change and bootstrap bound must equal its nominal value.
    check("all current bit intervals singleton", np.array_equal(lower, upper))
    check("all current task intervals singleton", np.all(possible.sum(axis=-1) == 1))
    metrics, denoms = {}, {}
    def put(name, block, value, denominator=1):
        metrics.setdefault(name, np.empty(64, dtype=float))[block] = value
        if name in denoms:
            assert denoms[name] == denominator
        denoms[name] = denominator
    mapping_full = np.empty((64, 2, 3)); mapping_edit = np.empty_like(mapping_full)
    full_contrast = np.empty((64, 2, 2)); edit_contrast = np.empty_like(full_contrast)
    change_stats = np.empty_like(mapping_full); primary_transport = np.zeros((64, 2)); primary_changes = np.zeros((64, 2))
    signed_changes = np.empty((64, 2, 2, 4), dtype=np.int8)
    error_rows = []
    for block in range(64):
        pair = int(bank["pair_index"][block]); ids = [2 * pair, 2 * pair + 1]; off = [u for u in range(4) if u not in ids]
        flat = [(c, s, u) for c in range(2) for s in range(4) for u in ids]
        tp = sum(int(bits[block,c,s,u] == 1) for c,s,u in flat if bank["expected_unit_bits"][block,c,s,u] == 1)
        tn = sum(int(bits[block,c,s,u] == 0) for c,s,u in flat if bank["expected_unit_bits"][block,c,s,u] == 0)
        put("primary_sensitivity",block,tp,2);put("primary_specificity",block,tn,14)
        put("primary_balanced_accuracy",block,(tp/2+tn/14)/2)
        patterns = [int(all(bits[block,c,s,u] == bank["expected_unit_bits"][block,c,s,u] for s in range(4) for u in ids)) for c in range(2)]
        put("positive_exact_pattern",block,patterns[0]);put("negative_exact_pattern",block,patterns[1])
        put("off_pair_false_positive_rate",block,sum(int(bits[block,c,s,u]) for c in range(2) for s in range(4) for u in off),16)
        for direction, states in [("original",(0,1)),("renamed",(2,3))]:
            entries = [(c,s,u) for c in range(2) for s in states for u in ids]
            put(direction+"_sensitivity",block,sum(int(bits[block,c,s,u]) for c,s,u in entries if bank["expected_unit_bits"][block,c,s,u] == 1))
            put(direction+"_specificity",block,sum(1-int(bits[block,c,s,u]) for c,s,u in entries if bank["expected_unit_bits"][block,c,s,u] == 0),7)
        for cohort, cohort_name in enumerate(("positive","negative")):
            row = [[int(v) for v in bits[block,cohort,s]] for s in range(4)]
            delta = [[row[1][u]-row[0][u] for u in range(4)],[row[3][u]-row[2][u] for u in range(4)]]
            signed_changes[block,cohort] = delta
            for mi,(mapping,permutation) in enumerate(MAPS.items()):
                discrepancies = [sum(row[s+2][u] != row[s][permutation[u]] for u in range(4))/4 for s in (0,1)]
                mapping_full[block,cohort,mi] = sum(discrepancies)/2
                mapping_edit[block,cohort,mi] = discrepancies[1]
                change_stats[block,cohort,mi] = sum(abs(delta[1][u]-delta[0][permutation[u]]) for u in range(4))/4
                put(mapping+"_"+cohort_name+"_endpoint_error",block,sum(discrepancies)/2)
                put(mapping+"_"+cohort_name+"_edited_endpoint_error",block,discrepancies[1])
                put(mapping+"_"+cohort_name+"_signed_change_discrepancy",block,change_stats[block,cohort,mi])
            for ci in range(2):
                full_contrast[block,cohort,ci] = mapping_full[block,cohort,ci+1]-mapping_full[block,cohort,0]
                edit_contrast[block,cohort,ci] = mapping_edit[block,cohort,ci+1]-mapping_edit[block,cohort,0]
            primary_transport[block,cohort] = sum(row[s+2][u] != row[s][MAPS["correct"][u]] for s in (0,1) for u in ids)
            primary_changes[block,cohort] = sum(delta[1][u] != delta[0][MAPS["correct"][u]] for u in ids)
        put("primary_transport_mismatch_rate",block,sum(primary_transport[block]),8)
        put("primary_response_change_mismatch_rate",block,sum(primary_changes[block]),4)
        for ci,name in enumerate(("identity","wrong")):
            put("edited_advantage_"+name,block,edit_contrast[block,0,ci])
            put("full_positive_advantage_"+name,block,full_contrast[block,0,ci])
        tgood = pred[block] == bank["labels"][block]
        task_tp = sum(int(tgood[c,s]) for c in range(2) for s in range(4) if bank["labels"][block,c,s] == 13)
        task_tn = sum(int(tgood[c,s]) for c in range(2) for s in range(4) if bank["labels"][block,c,s] == 14)
        put("task_sensitivity",block,task_tp,2);put("task_specificity",block,task_tn,6)
        put("task_balanced_accuracy",block,(task_tp/2+task_tn/6)/2);put("task_accuracy",block,task_tp+task_tn,8)
        for c,name in enumerate(("positive","negative")):
            quartet = int(all(tgood[c]))
            put("task_"+name+"_accuracy",block,sum(int(x) for x in tgood[c]),4)
            put("task_"+name+"_complete_quartet",block,quartet)
            put("joint_"+name+"_pattern_and_task",block,patterns[c]*quartet)
        put("task_positive_sensitivity",block,sum(int(tgood[0,s]) for s in (1,3)),2)
        put("task_positive_specificity",block,sum(int(tgood[0,s]) for s in (0,2)),2)
        put("task_negative_specificity",block,sum(int(x) for x in tgood[1]),4)
        for c,s,u in flat:
            if bits[block,c,s,u] != bank["expected_unit_bits"][block,c,s,u]:
                error_rows.append({"block":block,"pair":pair+1,"cohort":c,"state":s,"unit":UNITS[u],"post":float(native["post"][block,c,s,u]),
                                   "expected":int(bank["expected_unit_bits"][block,c,s,u]),"observed":int(bits[block,c,s,u]),"native_task_correct":bool(tgood[c,s])})
    for key,value in [("full_mapping",mapping_full),("edited_mapping",mapping_edit),("full_contrasts",full_contrast),
                      ("edited_contrasts",edit_contrast),("signed_change_discrepancy",change_stats),
                      ("primary_transport_mismatch",primary_transport),("primary_response_change_mismatch",primary_changes)]:
        equal("mapping singleton bounds " + key, np.repeat(value[...,None],3,axis=-1),saved_events[key])
    equal("signed unit changes",signed_changes,saved_events["signed_changes"])
    return metrics,denoms,pred,possible,error_rows


def bootstrap_and_criteria(metrics,denoms,bank,saved,analysis):
    names = saved["metric_names"].tolist()
    check("exact 47-metric inventory",len(names)==len(set(names))==47 and set(names)==set(metrics))
    scalar = np.stack([metrics[n] for n in names],axis=1)
    values = np.repeat(scalar[...,None],3,axis=-1)
    denominators = np.array([denoms[n] for n in names],dtype=float)
    equal("all block values and singleton bounds",values,saved["block_values"],1e-14)
    equal("all fixed per-block denominators",denominators,saved["denominators_per_block"])
    equal("fixed scope labels",["P1","P2","pooled"],saved["scope_names"])
    equal("fixed bound labels",["nominal","lower","upper"],saved["bound_order"])
    rates = scalar/denominators
    ids = [[i for i in range(64) if int(bank["pair_index"][i])==p] for p in range(2)]
    points = np.array([rates[ix].sum(axis=0)/32 for ix in ids])
    points = np.vstack((points,points.sum(axis=0)/2))
    point_triple = np.repeat(points[...,None],3,axis=-1)
    equal("all point metrics",point_triple,saved["point_rates"],1e-14)
    indices=arrays(STUDY/"bootstrap_indices.npz")["indices"]
    check("saved 2000 stratified draws shape/range",indices.shape==(2000,2,32) and indices.dtype==np.int16 and np.all((0<=indices)&(indices<32)))
    # Portable replay reuses frozen indices; it generates no RNG values.
    # The archived independent audit separately checked the exact RNG stream.
    boot=np.empty((2000,3,47))
    for draw in range(2000):
        for p in range(2):
            multiplicities=Counter(int(i) for i in indices[draw,p])
            boot[draw,p]=sum(count*rates[ids[p][i]] for i,count in multiplicities.items())/32
        boot[draw,2]=(boot[draw,0]+boot[draw,1])/2
    equal("every bootstrap statistic",np.repeat(boot[...,None],3,axis=-1),saved["bootstrap_rates"],1e-12)
    sorted_boot=np.sort(boot,axis=0)
    intervals=[]
    for probability in (0.025,0.975):
        position=probability*1999;left=int(position);fraction=position-left
        intervals.append(sorted_boot[left]*(1-fraction)+sorted_boot[left+1]*fraction)
    intervals=np.repeat(np.array(intervals)[...,None],3,axis=-1)
    equal("manual linear percentile interpolation",intervals,saved["interval_quantiles"],1e-12)
    for si,scope in enumerate(("P1","P2","pooled")):
        check("JSON metric inventory "+scope,set(analysis["metrics"][scope])==set(names))
        for mi,name in enumerate(names):
            record=analysis["metrics"][scope][name]
            equal("JSON points "+scope+"/"+name,record["point_nominal_lower_upper"],point_triple[si,mi],1e-14)
            equal("JSON lower CI "+scope+"/"+name,record["bootstrap_025_nominal_lower_upper"],intervals[0,si,mi],1e-12)
            equal("JSON upper CI "+scope+"/"+name,record["bootstrap_975_nominal_lower_upper"],intervals[1,si,mi],1e-12)
            check("JSON fixed denominator "+scope+"/"+name,record["aggregation_denominator"]==int(denoms[name]*(32 if si<2 else 64)))
    derived={}; directions={}
    for p,pname in enumerate(("P1","P2")):
        for name,cutoff,op in [("primary_sensitivity",58,">="),("primary_specificity",404,">="),("positive_exact_pattern",26,">="),("off_pair_false_positive_rate",51,"<=")]:
            value=float(sum(metrics[name][i] for i in ids[p]));derived[pname+"/"+name]=(value,cutoff,op)
        for comp in ("identity","wrong"):
            derived[pname+"/edited_advantage_"+comp]=(float(sum(metrics["edited_advantage_"+comp][i] for i in ids[p])/32),0,">")
        directions[pname]={}
        for direction in ("original","renamed"):
            tp=float(sum(metrics[direction+"_sensitivity"][i] for i in ids[p]));tn=float(sum(metrics[direction+"_specificity"][i] for i in ids[p]))
            status="PASS" if tp>=29 and tn>=202 else "CRITERION_NOT_MET"
            d=analysis["directional_criteria"][pname][direction]
            check("directional criterion "+pname+"/"+direction,d["status"]==status and d["positive_denominator"]==32 and d["negative_denominator"]==224)
            equal("direction TP "+pname+"/"+direction,d["conditions"]["true_positive"]["values_nominal_lower_upper"],[tp]*3)
            equal("direction TN "+pname+"/"+direction,d["conditions"]["true_negative"]["values_nominal_lower_upper"],[tn]*3)
            directions[pname][direction]={"TP":int(tp),"TN":int(tn),"status":status}
        localized=directions[pname]["original"]["status"]=="PASS" and directions[pname]["renamed"]["status"]=="CRITERION_NOT_MET"
        check("localized-transport interpretation "+pname,analysis["directional_criteria"][pname]["localized_transport_limitation"]==localized)
    for comp in ("identity","wrong"):
        derived["pooled/interval_lower_"+comp]=(float(intervals[0,2,names.index("edited_advantage_"+comp),0]),0,">")
    check("exact benchmark inventory",set(derived)==set(analysis["benchmark_conditions"]))
    failed=[]
    for key,(value,cutoff,op) in derived.items():
        passed=value>=cutoff if op==">=" else value<=cutoff if op=="<=" else value>cutoff
        record=analysis["benchmark_conditions"][key]
        equal("criterion value "+key,record["values_nominal_lower_upper"],[value]*3,1e-12)
        check("criterion rule "+key,record["cutoff"]==cutoff and record["comparison"]==op
              and all(record[flag]==passed for flag in ("nominal_pass","adverse_pass","favourable_pass")))
        if not passed:failed.append(key)
    check("overall finite-cohort criterion",analysis["status"]==("CRITERION_NOT_MET" if failed else "PASS") and analysis["nominal_benchmark_pass"]==(not failed))
    check("bootstrap receipt",analysis["bootstrap"]["draws"]==analysis["bootstrap"]["valid_draws"]==2000
          and analysis["bootstrap"]["seed"]==2026101605 and analysis["bootstrap"]["quantile_method"]=="linear"
          and analysis["bootstrap"]["indices_sha256"]==sha(STUDY/"bootstrap_indices.npz"))
    with (OUT/"recomputed_summaries.npz").open("xb") as handle:
        np.savez_compressed(handle,metric_names=np.array(names),block_values=values,point_rates=point_triple,interval_quantiles=intervals)
    return {"failed_criteria":failed,"directional_counts":directions,"pooled_mapping_advantage_interval":intervals[:,2,names.index("edited_advantage_identity"),0].tolist(),
            "study_draws_reused":2000,"computational_index_replay_draws":0,"additional_scientific_draws":0}


def affine(native,bank,possible):
    weights=arrays(STUDY/"readout_weights.npz")
    out,bias,unembed=(weights[k].astype(float) for k in ("W_out","b_out","W_U"))
    check("readout shapes",out.shape==(128,512) and bias.shape==(128,) and unembed.shape==(128,16))
    # Independently associate the real-arithmetic expression as one residual
    # vector and then unembed, rather than the study's three logit terms.
    hidden=native["full_post"].astype(float)@out.T+native["resid_mid"].astype(float)+bias
    reconstructed=hidden@unembed
    raw=native["logits"].astype(float)
    residual=reconstructed-raw
    saved=arrays(STUDY/"affine_reconstruction.npz")
    equal("independent affine logit reconstruction",reconstructed,saved["reconstructed_logits"],1e-10)
    equal("independent affine residuals",residual,saved["residuals"],1e-10)
    check("all 8192 native logits within declared parity",np.all(np.abs(residual)<=0.0001+0.00001*np.abs(raw)))
    coeff=np.empty((4,16),dtype=float)
    for u,unit in enumerate(UNITS):
        for token in range(16):
            coeff[u,token]=sum(out[h,unit]*unembed[h,token] for h in range(128))
    clamp=np.empty((64,2,2,2,16));contributions=np.empty((64,2,2,2));label=np.empty((64,2,2,2),dtype=int)
    for b in range(64):
        primary=[2*int(bank["pair_index"][b]),2*int(bank["pair_index"][b])+1]
        control=[u for u in range(4) if u not in primary]
        for c,d,w in itertools.product(range(2),range(2),range(2)):
            before,after=((0,1),(2,3))[d];selected=(primary,control)[w]
            change=np.array([sum((float(native["post"][b,c,after,u])-float(native["post"][b,c,before,u]))*coeff[u,t] for u in selected) for t in range(16)])
            clamp[b,c,d,w]=raw[b,c,after]-change
            contributions[b,c,d,w]=change[13]-change[14]
            label[b,c,d,w]=int(bank["labels"][b,c,after])
    pred,options,good=task_vectors(clamp,label)
    nativepred=native["logits"][:,:,[1,3]].argmax(axis=-1)
    nominal_change=pred!=nativepred[...,None]
    stable=nominal_change & (options.sum(axis=-1)==1) & (possible[:,:,[1,3]].sum(axis=-1)==1)[...,None]
    sorted_values=np.sort(clamp,axis=-1);gaps=sorted_values[...,-1]-sorted_values[...,-2]
    saved=arrays(STUDY/"affine_clamps.npz")
    for name,value,tol in [("logits",clamp,1e-10),("sat_minus_unsat_contribution",contributions,1e-10),
                           ("nominal_class",pred,0),("possible_classes",options,0),("correctness_bounds",good,0),
                           ("nominal_class_change",nominal_change,0),("numerically_stable_class_change",stable,0),("winning_gap",gaps,1e-10)]:
        equal("affine clamp "+name,value,saved[name],tol)
    receipt=readj(STUDY/"affine_accounting.json")
    check("secondary parity disposition",receipt["parity_pass"] and receipt["status"]=="PASS_SECONDARY_ACCOUNTING_ONLY")
    check("secondary residual summary",abs(receipt["max_abs_residual"]-float(np.abs(residual).max()))<1e-10)
    summaries={}
    for c,cohort in enumerate(("positive","negative")):
        for w,which in enumerate(("nominated_pair","other_fixed_pair")):
            oldcorrect=nativepred[:,c]==label[:,c,:,w];newcorrect=pred[:,c,:,w]==label[:,c,:,w]
            current={"edited_endpoints":128,"nominal_changes":int(nominal_change[:,c,:,w].sum()),"stable_changes":int(stable[:,c,:,w].sum()),
                     "correct_to_wrong":int((oldcorrect & ~newcorrect).sum()),"wrong_to_correct":int((~oldcorrect & newcorrect).sum()),
                     "clamp_nonanswer":int(sum(p not in (13,14) for p in pred[:,c,:,w].ravel())),
                     "mean_signed_sat_minus_unsat_contribution":float(contributions[:,c,:,w].mean())}
            key=cohort+"/"+which;summaries[key]=current
            check("affine summary "+key,all(abs(current[k]-receipt["summaries"][key][k])<1e-10 for k in current))
    return {"max_abs_native_residual":float(np.abs(residual).max()),"summaries":summaries,"clamp_ambiguous_rows":int((options.sum(axis=-1)>1).sum())}


def main():
    sha(__file__)
    counts=provenance()
    semantic=readj(OUT/"bank_replay.json")
    check("independent semantic replay passed",semantic["status"]=="PASS")
    bank=arrays(STUDY/"selected_bank.npz");native=arrays(STUDY/"native_outputs.npz");replay=arrays(STUDY/"replay_outputs.npz")
    selected=readj(STUDY/"selected_bank.json")
    replay_diffs=finite_native_and_replay(native,replay,selected)
    # Independently bind concatenated native/replay arrays to every saved batch.
    for directory,data in [("native_batches",native),("replay_batches",replay)]:
        for row,block in enumerate(data["block_ids"]):
            batch=arrays(STUDY/directory/("block_%03d.npz"%int(block)))
            for key in ("pre","post","full_post","resid_mid","logits"):
                equal(directory+"/"+str(int(block))+"/"+key,batch[key],data[key][row].reshape(8,-1))
    saved_events=arrays(STUDY/"event_bounds.npz");saved_metrics=arrays(STUDY/"bounded_metrics.npz");analysis=readj(STUDY/"analysis.json")
    metrics,denoms,pred,possible,error_rows=events_and_metrics(bank,native,saved_events)
    bootstrap=bootstrap_and_criteria(metrics,denoms,bank,saved_metrics,analysis)
    secondary=affine(native,bank,possible)
    check("no ambiguous native decisions reported",analysis["ambiguous_bit_positions"]==analysis["ambiguous_task_rows"]==0)
    check("nonanswer count full16",analysis["native_nonanswer_rows"]==int(sum(int(p) not in (13,14) for p in pred.ravel())))
    check("negative-control sensitivity undefined",analysis["negative_control_task_sensitivity"] is None)
    # CSV is a reader-facing export and must preserve all 2048 event rows.
    sha(STUDY/"input_unit_outcomes.csv")
    with (STUDY/"input_unit_outcomes.csv").open(newline="") as handle: rows=list(csv.DictReader(handle))
    check("complete per-input unit CSV",len(rows)==2048)
    seen=set()
    for row in rows:
        b=int(row["block"]);c=("positive","negative").index(row["cohort"]);s=("original","edited","renamed_original","renamed_edited").index(row["state"]);u=UNITS.index(int(row["unit"]));ix=(b,c,s,u)
        check("CSV row identity",ix not in seen);seen.add(ix)
        check("CSV exact values",float(row["pre"])==float(native["pre"][ix]) and float(row["post"])==float(native["post"][ix])
              and int(row["expected_bit"])==int(bank["expected_unit_bits"][ix]) and int(row["nominal_bit"])==int(saved_events["nominal_bits"][ix])
              and int(row["oracle_class"])==int(bank["labels"][b,c,s]) and int(row["native_class"])==int(pred[b,c,s])
              and [int(x) for x in row["possible_classes"].split()]==list(np.flatnonzero(possible[b,c,s])))
    writej("unit_event_errors.json",error_rows)
    # Verify every study input read by this audit remains byte-identical.
    changed=[]
    for path,record in list(inputs.items()):
        if presentation_identity(Path(path).read_bytes())!=record["sha256"]:changed.append(path)
    check("all inspected source files unchanged",not changed,changed)
    task_counts={"correct":int((pred==bank["labels"]).sum()),"total":512,
                 "SAT_correct":int(((pred==bank["labels"])&(bank["labels"]==13)).sum()),"SAT_total":128,
                 "UNSAT_correct":int(((pred==bank["labels"])&(bank["labels"]==14)).sum()),"UNSAT_total":384}
    receipt={"status":"PASS","scope":"Portable saved-array replay of the independent audit; no RNG regeneration, neural execution or new study draws",
             "contract_sha256":CONTRACT_SHA,"inputs":inputs,"checks":checks,"checks_passed":len(checks),
             "maximum_differences":differences,"native_journal_counts":counts,"native_replay_differences":replay_diffs,
             "bootstrap_replay":bootstrap,"secondary_affine":secondary,"task_counts":task_counts,"unit_event_errors":len(error_rows),
             "new_formulas":0,"new_neural_forwards":0,"fits":0,"additional_scientific_bootstrap_draws":0,
             "computational_bootstrap_index_replay_draws":0,"numpy":np.__version__,"elapsed_seconds":time.monotonic()-started,
             "limitations":["Exact RNG stream identity is not rerun here; the archived independent receipt records that check",
                            "Model-generated audit, same model family; oracle authored by this reviewer before construction",
                            "No neural/checkpoint execution or independently generated training/test cohort",
                            "No independent regeneration of proposal pool or selection ordering",
                            "Current bit/class intervals are singleton; unobserved ambiguous branches are not empirically validated",
                            "Numerical intervals are engineering sensitivity conventions, not proven error bounds",
                            "Clamps are affine saved-array calculations, not patched-native forwards or upstream mediation"]}
    writej("receipt.json",receipt)
    print(json.dumps({"status":"PASS","checks":len(checks),"task_counts":task_counts,"failed_criteria":bootstrap["failed_criteria"],"secondary":secondary["summaries"],"seconds":receipt["elapsed_seconds"]},indent=2))


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        writej("failure.json",{"status":"FAIL","exception":type(exc).__name__,"message":str(exc),"checks":checks,"inputs":inputs})
        raise
