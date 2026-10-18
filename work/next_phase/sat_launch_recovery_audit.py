"""Read-only saved-artifact audit; no production imports, formulas, RNG or inference."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parents[1]
OLD = WORK / "sat_renaming"
NEW = WORK / "sat_renaming_launch_recovery"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def load(p):
    with np.load(p, allow_pickle=False) as f: return {k: f[k].copy() for k in f.files}
def events(p): return [json.loads(line) for line in Path(p).read_text().splitlines()]
def instant(s): return dt.datetime.fromisoformat(s)

receipt = {"scope": "Independent saved-artifact/hash/array audit; zero new formulas, native forwards, fits, RNG calls or bootstrap draws.", "numpy": np.__version__}
hash_checks = {}
for root, names in [(OLD, ["freeze.json", "selected_data_freeze.json", "MANIFEST.json"]), (NEW, ["freeze.json", "selected_data_freeze.json", "MANIFEST.json"])]:
    for name in names:
        d = read(root/name)
        bad = [r["path"] for r in d["files"] if sha(root/r["path"]) != r["sha256"]]
        assert not bad, (str(root/name), bad)
        hash_checks[root.name+"/"+name] = {"sha256": sha(root/name), "verified_entries": len(d["files"]), "mismatches": bad}
recovery = read(NEW/"recovery_freeze.json")
for row in recovery["unchanged_copies"]:
    assert sha(NEW/row["path"]) == row["sha256"] == sha(OLD/row["path"])
assert sha(WORK/"recover_sat_renaming_launch.py") == sha(NEW/"recovery_driver.py") == recovery["driver_sha256"]
assert sha(WORK/"next_phase/sat_launch_recovery_admission.md") == recovery["admission_sha256"]
assert all(not (OLD/name).exists() for name in recovery["no_original_native_artifacts"])
original_execution = read(OLD/"execution.json")
assert original_execution["status"] == "UNRESOLVED_RUNTIME_FAILURE"
assert original_execution["batch_attempts"] == original_execution["completed_input_forwards"] == original_execution["conservative_input_forward_charge"] == 0
for name in ("execution.json", "execution_started.json", "execution_events.jsonl", "native_initial.log", "REPORT.md", "MANIFEST.json"):
    assert sha(OLD/name) == sha(NEW/"original_failure_provenance"/name)
receipt["hash_checks"] = hash_checks
receipt["byte_identical_recovery_copies"] = len(recovery["unchanged_copies"])
receipt["original_failure_preserved"] = True

ex = read(NEW/"execution.json")
timeline = events(NEW/"execution_events.jsonl")
attempts = [e for e in timeline if e["event"] == "STAGE_ATTEMPT"]
completions = [e for e in timeline if e["event"] == "STAGE_COMPLETE"]
assert len(attempts) == len(completions) == 3
assert [Path(e["command"][2]).name for e in attempts] == ["native.py", "native.py", "analyze.py"]
assert [e["command"][3] for e in attempts[:2]] == ["initial", "replay"]
assert all(e["returncode"] == 0 for e in completions)
assert all(e["command"][0] == recovery["native_executable_lexical"] for e in attempts[:2])
deadline = instant(recovery["original_native_deadline"])
freeze_time = instant(recovery["utc"])
times = [instant(e["utc"]) for e in timeline]
assert times == sorted(times)
assert freeze_time < times[0] < instant(completions[0]["utc"]) < instant(attempts[1]["utc"]) < instant(completions[1]["utc"]) < instant(attempts[2]["utc"]) < instant(completions[2]["utc"])
assert instant(completions[1]["utc"]) < deadline
assert completions[2]["elapsed_seconds"] < 300
assert all(e["timeout_seconds"] <= 600 for e in attempts[:2])
receipt["chronology"] = {"recovery_freeze_utc": recovery["utc"], "initial_attempt_utc": attempts[0]["utc"], "initial_complete_utc": completions[0]["utc"], "replay_complete_utc": completions[1]["utc"], "analysis_complete_utc": completions[2]["utc"], "original_deadline": recovery["original_native_deadline"], "seconds_remaining_after_replay": (deadline-instant(completions[1]["utc"])).total_seconds()}

combined_charge = combined_complete = 0
for stage, expected in [("initial", list(range(64))), ("replay", [0,1])]:
    rows = events(NEW/("native_"+stage+"_events.jsonl"))
    ba = [r for r in rows if r["event"] == "BATCH_ATTEMPT"]
    bc = [r for r in rows if r["event"] == "BATCH_COMPLETE"]
    assert [r["block_id"] for r in ba] == [r["block_id"] for r in bc] == expected
    assert [r["event"] for r in rows] == [x for _ in expected for x in ("BATCH_ATTEMPT", "BATCH_COMPLETE")]
    charge = sum(r["conservative_input_forward_charge"] for r in ba)
    completed = sum(r["input_forwards"] for r in bc)
    assert charge == completed == len(expected)*8
    combined_charge += charge; combined_complete += completed
assert combined_charge == combined_complete == ex["conservative_input_forward_charge"] == ex["completed_input_forwards"] == 528
assert ex["batch_attempts"] == 66 and ex["learner_fits"] == ex["new_proposals"] == 0
receipt["accounting"] = {"charged_input_forwards": combined_charge, "completed_input_forwards": combined_complete, "batches": 66, "new_proposals": 0, "fits": 0}

native, replay, bank = load(NEW/"native_outputs.npz"), load(NEW/"replay_outputs.npz"), load(NEW/"selected_bank.npz")
keys = ("pre", "post", "full_post", "resid_mid", "logits")
for label, data, directory in [("initial", native, "native_batches"), ("replay", replay, "replay_batches")]:
    for row, block in enumerate(data["block_ids"]):
        batch = load(NEW/directory/("block_%03d.npz"%block))
        for key in keys: assert np.array_equal(data[key][row], batch[key].reshape(2,4,-1))
for key in keys: assert np.array_equal(replay[key], native[key][[0,1]])
assert np.array_equal(native["post"], np.maximum(native["pre"],0))
assert np.array_equal(native["post"], native["full_post"][...,native["units"]])
bits = native["post"] >= .5
r = 1e-4 + 1e-5*np.abs(native["post"].astype(np.float64))
assert np.array_equal(bits, native["post"]-r>=.5)
assert np.array_equal(bits, native["post"]+r>=.5)
truth = bank["expected_unit_bits"].astype(bool)
prediction = native["logits"].argmax(axis=-1)
pair_results = {}
correct_perm = [1,0,3,2]; identity_perm = [0,1,2,3]; wrong_perm = [3,2,1,0]
edited_advantages = []
for p in (0,1):
    rows = np.flatnonzero(bank["pair_index"]==p); ids = [2*p,2*p+1]; off = [u for u in range(4) if u not in ids]
    b, t = bits[rows][...,ids], truth[rows][...,ids]
    good = b == t
    source, target = bits[rows,0,1], bits[rows,0,3]
    errors = [np.mean(target != source[:,perm],axis=-1) for perm in (correct_perm,identity_perm,wrong_perm)]
    advantages = np.stack([errors[1]-errors[0], errors[2]-errors[0]],axis=-1)
    edited_advantages.append(advantages)
    pair_results["P"+str(p+1)] = {"true_positive": int((b&t).sum()), "positive_denominator": int(t.sum()), "true_negative": int((~b&~t).sum()), "negative_denominator": int((~t).sum()), "positive_exact_patterns":int(good[:,0].all(axis=(1,2)).sum()), "negative_exact_patterns":int(good[:,1].all(axis=(1,2)).sum()), "off_pair_false_positives":int(bits[rows][...,off].sum()), "original_tp":int((b[:,:,:2]&t[:,:,:2]).sum()), "renamed_tp":int((b[:,:,2:]&t[:,:,2:]).sum()), "edited_advantages":advantages.mean(axis=0).tolist(), "task_correct":int((prediction[rows]==bank["labels"][rows]).sum())}
receipt["pair_results"] = pair_results
receipt["task_correct_total"] = int((prediction==bank["labels"]).sum())
receipt["saved_replay_bitwise_equal"] = True

# Reuse only existing draw indices; never create new resampling draws or call RNG.
indices = load(NEW/"bootstrap_indices.npz")["indices"]
assert indices.shape == (2000,2,32) and int(indices.min()) >= 0 and int(indices.max()) < 32
replayed = np.stack([edited_advantages[p][indices[:,p]].mean(axis=1) for p in (0,1)],axis=1).mean(axis=1)
interval = np.quantile(replayed,[.025,.975],axis=0,method="linear")
analysis = read(NEW/"analysis.json")
for i,name in enumerate(("identity","wrong")):
    metric = analysis["metrics"]["pooled"]["edited_advantage_"+name]
    assert np.allclose(interval[:,i], [metric["bootstrap_025_nominal_lower_upper"][0],metric["bootstrap_975_nominal_lower_upper"][0]],atol=0,rtol=0)
assert analysis["status"] == ex["status"] == "CRITERION_NOT_MET"
assert pair_results["P2"]["true_positive"] < 58 and pair_results["P2"]["positive_exact_patterns"] < 26
assert not analysis["directional_criteria"]["P2"]["localized_transport_limitation"]
receipt["saved_index_contrast_quantiles"] = interval.tolist()
receipt["new_bootstrap_draws"] = 0

clamps = load(NEW/"affine_clamps.npz")
saved_clamp_pred = clamps["logits"].argmax(axis=-1)
assert np.array_equal(saved_clamp_pred, clamps["nominal_class"])
native_edited = prediction[:,:,[1,3]]
changes = saved_clamp_pred != native_edited[...,None]
assert np.array_equal(changes,clamps["nominal_class_change"])
receipt["clamp_changes"] = {"positive_nominated_pair":int(changes[:,0,:,0].sum()), "positive_other_pair":int(changes[:,0,:,1].sum()), "negative_nominated_pair":int(changes[:,1,:,0].sum()), "negative_other_pair":int(changes[:,1,:,1].sum())}
receipt["status"] = "AUDIT_PASS_NO_CRITICAL_FINDING"
hash_names = ["recovery_freeze.json", "recovery_driver.py", "MANIFEST.json", "execution.json", "native_initial.json", "native_replay.json", "native_outputs.npz", "replay_outputs.npz", "analysis.json", "bootstrap_indices.npz", "affine_accounting.json", "affine_clamps.npz", "REPORT.md"]
receipt["exact_inputs"] = {name:sha(NEW/name) for name in hash_names}
out = WORK/"next_phase/sat_launch_recovery_postrun_receipt.json"
with out.open("x") as f: json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(receipt,indent=2,sort_keys=True))
