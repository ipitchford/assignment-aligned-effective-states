"""Frozen scalar readiness, native transport, controls and clustered uncertainty."""
import argparse
import json
import platform
import time
import warnings
import numpy as np
import scipy
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from protocol import *

assert (np.__version__,scipy.__version__,sklearn.__version__)==("2.4.6","1.17.1","1.9.0"), "Pinned analysis runtime drift"


def versions():return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)


def auc(y,score):
    return float(roc_auc_score(y,score)) if len(y) and np.any(y) and np.any(~y) else None


def summaries(score,post,q,scope):
    out={}
    for name,mask in [("zero",q==0),("one",q==1),("many",q>=2)]:
        selected=scope&mask;s=score[selected];p=post[selected]
        out[name]=dict(n=len(s),clipped=int((s<0).sum()),zero_preactivation=int((s==0).sum()),native_high=int((p>=.5).sum()),mean_pre=float(s.mean()) if len(s) else None,min_pre=float(s.min()) if len(s) else None,max_pre=float(s.max()) if len(s) else None,mean_post=float(p.mean()) if len(p) else None)
    return out


def stage_a():
    frozen();start=time.monotonic()
    if (ROOT/"stage_a.json").exists():raise RuntimeError("Stage A already analyzed")
    rows=mapping();native=np.load(ROOT/"original_native.npz",allow_pickle=False)
    bank=np.load(PILOT/"pilot_bank.npz",allow_pickle=False)
    fit=np.arange(4096)<3072;assessment=~fit
    prebits=np.zeros((4096,17),dtype=bool);postbits=prebits.copy();distances=np.full((4096,17),np.inf,dtype=np.float32)
    records=[]
    for ui,row in enumerate(rows):
        col=int(np.flatnonzero(PUBLISHED_UNITS==row["unit"])[0])
        score=native["pre"][:,col];post=native["post"][:,col];q=bank["counts"][:,row["truth_table_index"]]
        clipped=(q>=1)&(score<0);mask=fit&clipped
        pre_rule=threshold_rule(score[mask],q[mask]==1);post_rule=threshold_rule(post[mask],q[mask]==1)
        prebits[:,ui]=apply_rule(score,pre_rule);postbits[:,ui]=apply_rule(post,post_rule)
        if pre_rule["kind"]=="threshold":distances[:,ui]=np.abs(score-np.float32(pre_rule["threshold"]))
        record=dict(unit=row["unit"],assignment=row["assignment"],assignment_index=row["truth_table_index"],pre_rule=pre_rule,post_rule=post_rule)
        for name,scope in [("fit",fit),("assessment",assessment),("overall",np.ones(4096,dtype=bool))]:
            selected=scope&clipped;y=q[selected]==1
            record[name]=dict(origin_count=int(scope.sum()),q_positive_count=int((scope&(q>=1)).sum()),clipped_count=int(selected.sum()),clipped_coverage_all=float(selected.sum()/scope.sum()),clipped_coverage_q_positive=float(selected.sum()/(scope&(q>=1)).sum()) if (scope&(q>=1)).sum() else None,q_one=int(y.sum()),q_many=int((~y).sum()),pre_auroc=auc(y,score[selected]),post_auroc=auc(y,post[selected]),pre_threshold_metrics=support_metrics(y,prebits[selected,ui]),post_threshold_metrics=support_metrics(y,postbits[selected,ui]),unconditioned=summaries(score,post,q,scope))
        records.append(record)
    aggregate={}
    for scope in ["fit","assessment"]:
        aggregate[scope]={key:assignment_macro([r[scope][key] for r in records],rows) for key in ["pre_auroc","post_auroc"]}
        for key in ["pre_threshold_metrics","post_threshold_metrics"]:
            aggregate[scope][key+"_balanced_accuracy"]=assignment_macro([r[scope][key]["balanced_accuracy"] for r in records],rows)
    au=aggregate["assessment"]["pre_auroc"]
    supported=all(r["assessment"]["q_one"]>=20 and r["assessment"]["q_many"]>=20 for r in records)
    count=sum(v is not None and v>.60 for v in au["assignments"].values())
    conditions=dict(all_source_pairs_assessment_20_per_q_class=supported,macro_auroc_at_least_point65=au["macro"] is not None and au["macro"]>=.65,at_least_12_assignments_above_point60=count>=12)
    gate=dict(pass_=all(conditions.values()),conditions=conditions,macro_auroc=au["macro"],assignments_above_point60=count)
    gate["pass"]=gate.pop("pass_")
    np.savez_compressed(ROOT/"readiness.npz",pre_bits=prebits,post_bits=postbits,pre_threshold_distances=distances,units=np.array(UNITS))
    write_json("stage_a.json",dict(status="PASS" if gate["pass"] else "STOP_STAGE_A",scope="opened development origins; fixed score direction; not protected confirmation",gate=gate,units=records,aggregate=aggregate,versions=versions(),elapsed_seconds=time.monotonic()-start,freeze_sha256=sha(ROOT/"freeze.json")))
    print(json.dumps(dict(stage="A",gate=gate)))


def interaction(d,q,mask,weights=None):
    weights=np.ones(len(d),dtype=np.int64) if weights is None else weights
    one=mask&(q==1);many=mask&(q>=2)
    n1=int(weights[one].sum());nm=int(weights[many].sum())
    s1=int((d[one]*weights[one]).sum());sm=int((d[many]*weights[many]).sum())
    m1=s1/n1 if n1 else None;mm=sm/nm if nm else None
    return dict(q_one_n=n1,q_many_n=nm,q_one_sum_D=s1,q_many_sum_D=sm,q_one_mean_D=m1,q_many_mean_D=mm,interaction=m1-mm if m1 is not None and mm is not None else None,negative_D_cases=int(weights[mask&(d<0)].sum()),positive_D_cases=int(weights[mask&(d>0)].sum()),zero_D_cases=int(weights[mask&(d==0)].sum()))


def clustered_bootstrap(queries,clipped,y,pred,d,q,rows):
    assessment=np.array([r["split"]=="assessment" for r in queries]);origin=np.array([r["origin_index"] for r in queries]);ui=np.array([r["unit_index"] for r in queries])
    origin_ids=np.unique(origin[assessment]);lookup={int(v):i for i,v in enumerate(origin_ids)}
    cm=np.zeros((len(origin_ids),17,len(MODELS),4),dtype=np.int64)
    effect=np.zeros((len(origin_ids),17,4),dtype=np.int64)
    valid=np.ones((17,len(MODELS)),dtype=bool)
    for k in np.flatnonzero(assessment&clipped):
        oi=lookup[int(origin[k])];u=ui[k]
        for mi in range(len(MODELS)):
            p=pred[k,:,mi]
            if not np.isfinite(p).all():valid[u,mi]=False;continue
            p=p.astype(bool);truth=y[k]
            cm[oi,u,mi]+=np.array([(truth&p).sum(),((~truth)&(~p)).sum(),((~truth)&p).sum(),(truth&(~p)).sum()])
        if q[k]==1:effect[oi,u,:2]+=np.array([1,d[k]])
        else:effect[oi,u,2:]+=np.array([1,d[k]])
    rng=np.random.default_rng(2026101521)
    weights=np.array([np.bincount(rng.integers(0,len(origin_ids),size=len(origin_ids)),minlength=len(origin_ids)) for _ in range(2000)],dtype=np.int16)
    totals=(weights.astype(np.float64)@cm.reshape(len(origin_ids),-1).astype(np.float64)).reshape(2000,17,len(MODELS),4)
    effects=(weights.astype(np.float64)@effect.reshape(len(origin_ids),-1).astype(np.float64)).reshape(2000,17,4)
    positive=totals[:,:,:,0]+totals[:,:,:,3];negative=totals[:,:,:,1]+totals[:,:,:,2]
    with np.errstate(divide="ignore",invalid="ignore"):
        ba=(totals[:,:,:,0]/positive+totals[:,:,:,1]/negative)/2
        delta=effects[:,:,1]/effects[:,:,0]-effects[:,:,3]/effects[:,:,2]
    ba[:,~valid]=np.nan
    assignment_names=sorted(set(r["assignment"] for r in rows))
    ab=np.stack([ba[:,[i for i,r in enumerate(rows) if r["assignment"]==a],:].mean(axis=1) for a in assignment_names],axis=1)
    ad=np.stack([delta[:,[i for i,r in enumerate(rows) if r["assignment"]==a]].mean(axis=1) for a in assignment_names],axis=1)
    macro_ba=ab.mean(axis=1);macro_delta=ad.mean(axis=1)
    intervals=dict(native_balanced_accuracy={name:percentile_interval(macro_ba[:,mi]) for mi,name in enumerate(MODELS)},interaction=percentile_interval(macro_delta),structured_minus_control={name:percentile_interval(macro_ba[:,0]-macro_ba[:,mi]) for mi,name in enumerate(MODELS) if name not in ["structured","symbolic_oracle"]},per_unit={str(row["unit"]):dict(native_balanced_accuracy={name:percentile_interval(ba[:,i,mi]) for mi,name in enumerate(MODELS)},interaction=percentile_interval(delta[:,i])) for i,row in enumerate(rows)})
    np.savez_compressed(ROOT/"bootstrap_raw.npz",origin_ids=origin_ids,origin_multiplicities=weights,per_unit_balanced_accuracy=ba,per_unit_interaction=delta,assignment_balanced_accuracy=ab,assignment_interaction=ad,macro_balanced_accuracy=macro_ba,macro_interaction=macro_delta,model_names=np.array(MODELS),assignments=np.array(assignment_names))
    return dict(scope="descriptive development uncertainty; original-origin cluster resampling; all2000draws retained",seed=2026101521,distinct_assessment_origins=len(origin_ids),intervals=intervals)


def stage_b():
    frozen();start=time.monotonic();a=read_json("stage_a.json")
    if not a["gate"]["pass"]:raise RuntimeError("Stage A gate failed; Stage B forbidden")
    if (ROOT/"stage_b.json").exists():raise RuntimeError("Stage B already analyzed")
    rows=mapping();queries=read_json("selected_queries.json")["rows"];n=len(queries)
    original=np.load(ROOT/"original_native.npz",allow_pickle=False);edited=np.load(ROOT/"edited_native.npz",allow_pickle=False)
    ready=np.load(ROOT/"readiness.npz",allow_pickle=False);token_map=np.load(ROOT/"query_tokens.npz",allow_pickle=False)["query_action_to_unique"]
    ui=np.array([r["unit_index"] for r in queries]);origins=np.array([r["origin_index"] for r in queries]);q=np.array([r["original_q"] for r in queries]);assignment=np.array([r["assignment_index"] for r in queries]);split=np.array([r["split"] for r in queries]);old=np.array([r["old_clause"] for r in queries]);slot=np.array([r["slot"] for r in queries]);new=np.array([[r["repair_clause"],r["neutral_clause"]] for r in queries])
    columns=np.array([int(np.flatnonzero(PUBLISHED_UNITS==u)[0]) for u in UNITS])[ui]
    pre=original["pre"][origins,columns];clipped=(q>=1)&(pre<0)
    native_pre=edited["pre"][token_map,columns[:,None]];native_post=edited["post"][token_map,columns[:,None]]
    y=native_post>=.5;d=y[:,0].astype(np.int8)-y[:,1].astype(np.int8)
    effect=(~CLAUSE_TRUTH[old,assignment]).astype(int)[:,None]-(~CLAUSE_TRUTH[new,assignment[:,None]]).astype(int)
    assert np.array_equal(effect,np.tile([1,0],(n,1)))
    next_ui=(ui+1)%17;next_assignment=np.array([r["truth_table_index"] for r in rows])[next_ui]
    wrong_effect=(~CLAUSE_TRUTH[old,next_assignment]).astype(int)[:,None]-(~CLAUSE_TRUTH[new,next_assignment[:,None]]).astype(int)
    own=ready["pre_bits"][origins,ui];wrong=ready["pre_bits"][origins,next_ui];post_ready=ready["post_bits"][origins,ui]
    pred=np.full((n,2,len(MODELS)),np.nan,dtype=np.float64)
    pred[:,:,0]=own[:,None]&(effect==1)
    pred[:,:,1]=post_ready[:,None]&(effect==1)
    pred[:,:,2]=wrong[:,None]&(effect==1)
    pred[:,:,3]=own[:,None]&(wrong_effect==1)
    pred[:,:,5]=(q[:,None]==1)&(effect==1)
    contexts=np.stack([np.repeat(slot,2),np.repeat(old,2),new.reshape(-1)],axis=1).reshape(n,2,3)
    probes=[]
    for i,row in enumerate(rows):
        fit=(ui==i)&(split=="fit")&clipped;all_rows=ui==i
        target=y[fit].reshape(-1);unique=np.unique(target)
        probe=dict(unit=row["unit"],fit_rows=len(target),positive=int(target.sum()),negative=int((~target).sum()),warnings=[])
        if not len(unique):probe.update(kind="unestimable",parameter_count=0,reason="no clipped fit rows")
        elif len(unique)==1:
            pred[all_rows,:,4]=bool(unique[0]);probe.update(kind="constant",value=bool(unique[0]),parameter_count=0,constant_bits=1)
        else:
            estimator=LogisticRegression(C=1.0,class_weight="balanced",solver="lbfgs",max_iter=1000)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                estimator.fit(context_design(contexts[fit].reshape(-1,3)),target)
            pred[all_rows,:,4]=estimator.predict(context_design(contexts[all_rows].reshape(-1,3))).reshape(-1,2)
            probe.update(kind="logistic",parameter_count=int(estimator.coef_.size+estimator.intercept_.size),coefficient_dtype=str(estimator.coef_.dtype),coefficients=estimator.coef_.tolist(),intercept=estimator.intercept_.tolist(),classes=estimator.classes_.astype(int).tolist(),iterations=estimator.n_iter_.tolist(),warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in observed])
        probes.append(probe)
    unit_results=[]
    for i,row in enumerate(rows):
        record=dict(unit=row["unit"],assignment=row["assignment"])
        for name,scope in [("primary_assessment",(split=="assessment")&clipped),("all_selected_assessment",split=="assessment"),("primary_fit",(split=="fit")&clipped),("all_selected_fit",split=="fit")]:
            mask=(ui==i)&scope
            record[name]=dict(paired_origins=int(mask.sum()),clipped_paired_origins=int((mask&clipped).sum()),models={model:support_metrics(y[mask].reshape(-1),pred[mask,:,mi].reshape(-1)) for mi,model in enumerate(MODELS)},effect=interaction(d,q,mask))
        unit_results.append(record)
    aggregate={}
    for scope in ["primary_assessment","all_selected_assessment","primary_fit","all_selected_fit"]:
        aggregate[scope]=dict(models={model:assignment_macro([u[scope]["models"][model]["balanced_accuracy"] for u in unit_results],rows) for model in MODELS},effect={name:assignment_macro([u[scope]["effect"][name] for u in unit_results],rows) for name in ["q_one_mean_D","q_many_mean_D","interaction"]})
    primary=aggregate["primary_assessment"];native_ba=primary["models"]["structured"]["macro"];inter=primary["effect"]["interaction"]
    gaps={name:(native_ba-primary["models"][name]["macro"]) if native_ba is not None and primary["models"][name]["macro"] is not None else None for name in ["post_score","mismatched_unit","mismatched_assignment","context_only"]}
    coverage=all(u["primary_assessment"]["effect"]["q_one_n"]>=10 and u["primary_assessment"]["effect"]["q_many_n"]>=10 for u in unit_results)
    positive_assignments=sum(v is not None and v>0 for v in inter["assignments"].values())
    conditions=dict(all_source_pairs_10_clipped_assessment_per_q_class=coverage,macro_interaction_at_least_point25=inter["macro"] is not None and inter["macro"]>=.25,structured_native_balanced_accuracy_at_least_point70=native_ba is not None and native_ba>=.70,at_least_12_positive_assignment_interactions=positive_assignments>=12)
    for name,gap in gaps.items():conditions["exceeds_"+name]=gap is not None and gap>=(.05 if name=="context_only" else .10)
    reasons=[]
    if not coverage:reasons.append("coverage")
    if inter["macro"] is None or native_ba is None or any(v is None for v in gaps.values()):reasons.append("insufficient class support")
    if not conditions["macro_interaction_at_least_point25"] or not conditions["structured_native_balanced_accuracy_at_least_point70"]:reasons.append("native transport")
    if not conditions["at_least_12_positive_assignment_interactions"] or not all(conditions["exceeds_"+n] for n in ["post_score","mismatched_unit","mismatched_assignment"]):reasons.append("specificity")
    if not conditions["exceeds_context_only"]:reasons.append("context redundancy")
    bootstrap=clustered_bootstrap(queries,clipped,y,pred,d,q,rows)
    np.savez_compressed(ROOT/"stage_b_raw.npz",query_id=np.arange(n),unit_index=ui,origin_index=origins,split=split,original_q=q,clipped=clipped,original_pre=pre,native_edited_pre=native_pre,native_edited_post=native_post,native_high=y,relu_crossing=native_pre>0,repair_minus_neutral=d,action_effect=effect,mismatched_assignment_effect=wrong_effect,predictions=pred,model_names=np.array(MODELS),semantic_sat=np.array([[r["repair_sat"],r["neutral_sat"]] for r in queries]),edited_logits=edited["logits"][token_map],original_logits=original["logits"][origins],context_rows=contexts)
    write_json("stage_b.json",dict(status="PROMISING_DEVELOPMENT_ONLY" if all(conditions.values()) else "STOP_STAGE_B",scope="fixed native-unit prospective transport; not a global SAT repair or protected confirmation",gate=dict(pass_=all(conditions.values()),conditions=conditions,macro_structured_native_balanced_accuracy=native_ba,macro_interaction=inter["macro"],positive_assignments=positive_assignments,gaps=gaps,failure_categories=reasons),units=unit_results,aggregate=aggregate,context_probes=probes,bootstrap=bootstrap,versions=versions(),elapsed_seconds=time.monotonic()-start,freeze_sha256=sha(ROOT/"freeze.json")))
    result=read_json("stage_b.json");result["gate"]["pass"]=result["gate"].pop("pass_");write_json("stage_b.json",result)
    print(json.dumps(dict(stage="B",gate=result["gate"])))


def finalize():
    frozen();a=read_json("stage_a.json");b=read_json("stage_b.json") if (ROOT/"stage_b.json").exists() else None
    replay=read_json("replay.json");queries=read_json("selected_queries.json")
    original=read_json("original_extraction.json");edited=read_json("edited_extraction.json") if b else None
    forwards=original["forwards"]+(edited["forwards"] if edited else 0)+replay["total_forwards"]
    assert forwards<=8464
    params=sum(p["parameter_count"] for p in b["context_probes"]) if b else 0
    ledger=dict(state_bits=51,state_components=dict(published_bits=34,readiness_bits=17),action_fixed_bits=16,action_packed_bits=14,state_plus_action_fixed_bits=67,each_float32_34_value_report_bits=1088,pre_and_post_archive_bits_per_origin=2176,pre_threshold_static_float32_bits=17*32,post_control_threshold_static_float32_bits=17*32,threshold_rule_tag_bits_per_rule=2,source_unit_assignment_pairs_static_bits=17*(9+5),published_unit_id_list_static_bits=34*9,clause_truth_lookup_static_bits=40*32,context_probe_parameter_count=params,context_probe_float64_parameter_bits=params*64,context_probe_capacity_upper_parameters=17*91,context_probe_parameters_charged_separately=True,original_model_parameters=403840,checkpoint_bytes=(PILOT/"upstream/models/layers_2_heads_1_4_1M/900.pth").stat().st_size(),extraction_execution="unchanged two-block native checkpoint, CPU float32, two threads",structured_execution="one frozen comparison bit, two clause-falsification lookups, one action-effect equality and Boolean AND per nominated unit",context_execution="90-dimensional one-hot action design and fitted logistic dot product/intercept per nonconstant control",original_forwards=original["forwards"],selected_edit_rows_before_dedup=queries["edited_rows_before_dedup"],edited_forwards=edited["forwards"] if edited else 0,replay_forwards=replay["total_forwards"],total_native_forwards=forwards,ceiling=8464,semantic_comparators=dict(ordered_clauses_fixed_bits=60,ordered_clauses_packed_bits=54,clause_multiset_packed_bits=33,capped_counts_fixed_bits=64,capped_counts_packed_bits=51,computation="unranking/clause evaluation/SAT computation must be charged; sequence-sensitive native outputs not preserved"))
    write_json("cost_ledger.json",ledger)
    summary=dict(status="STOPPED_AFTER_CONTRACT",stage_a_gate=a["gate"],stage_b_gate=b["gate"] if b else None,replay_status=replay["status"],new_formulas=0,protected_cases_accessed=0,pilot_manifest_unchanged=True,resource_ledger=ledger,prior_pilot_postrelu_float_decision_success=.62265625,prior_pilot_bits_decision_success=.62451171875,no_global_sat_repair_claim=True)
    write_json("summary.json",summary)
    text="# Fixed-score native transport: contracted development result\n\n"
    text+=f"Stage A gate: **{'PASS' if a['gate']['pass'] else 'FAIL'}**. Assignment-weighted negative-score AUROC: {a['gate']['macro_auroc']}; {a['gate']['assignments_above_point60']}/16 assignments exceed 0.60.\n\n"
    if b:
        g=b["gate"];text+=f"Stage B admission gate: **{'PASS' if g['pass'] else 'FAIL'}**. Structured native balanced accuracy: {g['macro_structured_native_balanced_accuracy']}; repair-minus-neutral q-class interaction: {g['macro_interaction']}; positive assignment interactions: {g['positive_assignments']}/16.\n\n"
        text+="Failures: "+(", ".join(g["failure_categories"]) or "none at the registered development thresholds")+".\n\n"
        text+="| Forecast | Assignment-weighted native balanced accuracy |\n|---|---:|\n"
        for name in MODELS:text+=f"| {name} | {b['aggregate']['primary_assessment']['models'][name]['macro']} |\n"
        text+="\nAll unit-level supports, missing-class metrics, fit warnings, selected unclipped cases, paired effects and 2,000 origin-cluster bootstrap draws are retained. Intervals are descriptive development uncertainty; missing metrics propagate and draws are never replenished.\n\n"
    else:text+="Stage B was not run because its gate failed. No native edited outcomes were extracted.\n\n"
    text+=f"Separate-process numerical replay: **{replay['status']}**. Native forwards used: {forwards}/8464. The original pilot remains unchanged.\n\n"
    text+="This is a source-nominated, opened-data development assay of 17 units representing 16 assignments. It does not repair the earlier witness, establish a global SAT decoder, or show that negative values mediate the current output through ReLU. The prior post-ReLU magnitude pilot remains negative for decision improvement. The 51-state-bit plus 16-action-bit ledger is diagnostic, with source mappings, fitted constants, model extraction and context-probe parameters charged separately.\n\n"
    text+="The contract, selected queries and complete implementation were frozen before preactivation extraction. No new origin formulas, feature search, decoder search, threshold-direction reversal or protected evaluation occurred. Execution stops here.\n"
    (ROOT/"REPORT.md").write_text(text)
    print(json.dumps(summary))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["a","b","finalize"]);args=parser.parse_args()
    {"a":stage_a,"b":stage_b,"finalize":finalize}[args.stage]()
