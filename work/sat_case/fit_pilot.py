"""Fixed-capacity decoder comparison on development origins only."""
import json
import platform
import time
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,log_loss,confusion_matrix
from pilot_common import *

REPORT_NAMES=["context_only","published_bits34","published_float34","random_bits34","random_float34","truth_table32"]


def report_matrix(name, bank):
    if name=="context_only": return np.empty((4096,0),dtype=np.float32)
    if name=="truth_table32": return bank["truth_tables"].astype(np.float32)
    units=bank["published_units"] if name.startswith("published") else bank["random_units"]
    values=bank["activations"][:,units]
    return (values>=0.5).astype(np.float32) if "bits" in name else values.astype(np.float32)


def design(report,rows,clauses):
    result=np.zeros((len(rows),report.shape[1]+90),dtype=np.float32)
    n=report.shape[1]
    result[:,:n]=report[rows[:,0]]
    k=np.arange(len(rows))
    result[k,n+rows[:,1]]=1
    result[k,n+10+clauses[rows[:,0],rows[:,1]]]=1
    result[k,n+50+rows[:,2]]=1
    return result


def rowsets(clauses):
    rng=np.random.default_rng(2026101513)
    fit=[];assessment=[]
    for i in range(3072):
        for slot in range(10):
            choices=np.delete(np.arange(40),clauses[i,slot])
            for new in rng.choice(choices,size=2,replace=False): fit.append((i,slot,int(new)))
    for i in range(3072,4096):
        for slot in range(10):
            for new in range(40):
                if new!=clauses[i,slot]: assessment.append((i,slot,new))
    fit=np.array(fit,dtype=np.int32);assessment=np.array(assessment,dtype=np.int32)
    assert fit.shape==(61440,3) and assessment.shape==(399360,3)
    assert np.all(clauses[fit[:,0],fit[:,1]]!=fit[:,2])
    assert np.all(clauses[assessment[:,0],assessment[:,1]]!=assessment[:,2])
    return fit,assessment


def finite_collision_bound(report,clauses,post_sat):
    # Empirical minimum error of ANY deterministic rowwise decoder on this bank.
    if report.shape[1]:
        _,codes=np.unique(report,axis=0,return_inverse=True)
    else:
        codes=np.zeros(len(report),dtype=int)
    unavoidable=0;conflicting_groups=0;groups=0
    for slot in range(10):
        for old in range(40):
            selected=np.flatnonzero(clauses[:,slot]==old)
            if not len(selected): continue
            _,inverse=np.unique(codes[selected],return_inverse=True)
            size=np.bincount(inverse)
            for new in range(40):
                if new==old: continue
                positive=np.bincount(inverse,weights=post_sat[selected,slot,new],minlength=len(size)).astype(int)
                negative=size-positive
                unavoidable+=int(np.minimum(positive,negative).sum())
                conflicting_groups+=int(((positive>0)&(negative>0)).sum())
                groups+=len(size)
    return dict(unavoidable_errors=unavoidable,total_rows=len(clauses)*390,min_empirical_error=unavoidable/(len(clauses)*390),conflicting_report_action_groups=conflicting_groups,total_report_action_groups=groups,scope="finite entire development bank only; no distributional lower-bound claim")


def main():
    check_contract()
    if (ROOT/"predictive_metrics.json").exists(): raise RuntimeError("Refusing to overwrite completed pilot fit")
    started=time.monotonic()
    bank=np.load(ROOT/"pilot_bank.npz",allow_pickle=False)
    clauses=bank["clauses"];post_sat=bank["post_sat"]
    fit,assessment=rowsets(clauses)
    yfit=post_sat[fit[:,0],fit[:,1],fit[:,2]]
    y=post_sat[assessment[:,0],assessment[:,1],assessment[:,2]]
    newmenu=assessment[:,2].reshape(1024,10,39)
    unsatmenu=(~y).reshape(1024,10,39)
    reachable=unsatmenu.any(axis=-1)
    probabilities=[];selected=[];selection_p=[];per_origin_error=[];per_origin_success=[];metrics={}
    settings=dict(max_iter=100,max_leaf_nodes=15,learning_rate=0.08,l2_regularization=1.0,max_bins=63,early_stopping=False,random_state=2026101514)
    for name in REPORT_NAMES:
        phase=time.monotonic()
        report=report_matrix(name,bank)
        decoder=HistGradientBoostingClassifier(**settings)
        decoder.fit(design(report,fit,clauses),yfit)
        p=[]
        for offset in range(0,len(assessment),8192):
            p.append(decoder.predict_proba(design(report,assessment[offset:offset+8192],clauses))[:,int(np.flatnonzero(decoder.classes_==True)[0])])
        p=np.concatenate(p)
        assert np.isfinite(p).all()
        predicted=p>=0.5
        err=(predicted!=y).reshape(1024,390).mean(axis=1)
        menu_p=(1-p).reshape(1024,10,39)
        chosen=np.argmax(menu_p,axis=-1)
        success=np.take_along_axis(unsatmenu,chosen[:,:,None],axis=-1)[:,:,0]
        confidence=np.take_along_axis(menu_p,chosen[:,:,None],axis=-1)[:,:,0]
        choices=np.take_along_axis(newmenu,chosen[:,:,None],axis=-1)[:,:,0]
        accepted=confidence>=0.5
        origin_success=success.mean(axis=1)
        entry=dict(response=dict(accuracy=float(accuracy_score(y,predicted)),balanced_accuracy=float(balanced_accuracy_score(y,predicted)),auroc=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p,labels=[False,True])),confusion_unsat_sat=confusion_matrix(y,predicted,labels=[False,True]).tolist()),decision=dict(total_menus=int(success.size),reachable_menus=int(reachable.sum()),success_all_menus=float(success.mean()),success_when_reachable=float(success[reachable].mean()),coverage_at_half=float(accepted.mean()),successful_accepted_menus=int((accepted&success).sum()),success_among_accepted=float(success[accepted].mean()) if accepted.any() else None),input_report_dimensions=report.shape[1],elapsed_seconds=time.monotonic()-phase)
        if "float" not in name: entry["finite_bank_information_bound"]=finite_collision_bound(report,clauses,post_sat)
        probabilities.append(p.astype(np.float32));selected.append(choices);selection_p.append(confidence);per_origin_error.append(err);per_origin_success.append(origin_success)
        metrics[name]=entry
        print(json.dumps(dict(report=name,elapsed_seconds=entry["elapsed_seconds"],response=entry["response"],decision=entry["decision"])),flush=True)
    errors=np.array(per_origin_error);successes=np.array(per_origin_success)
    differences={}
    for left,right in [("published_float34","published_bits34"),("published_float34","random_float34"),("published_bits34","random_bits34"),("published_float34","truth_table32")]:
        a=REPORT_NAMES.index(left);b=REPORT_NAMES.index(right)
        delta=errors[b]-errors[a]
        decision_delta=successes[a]-successes[b]
        differences[left+"_minus_"+right]=dict(mean_response_accuracy_gain=float(delta.mean()),origin_error_improved=int((delta>0).sum()),origin_error_worsened=int((delta<0).sum()),origin_error_equal=int((delta==0).sum()),mean_decision_success_gain=float(decision_delta.mean()))
    receipt=dict(status="PASS",contract_sha256=CONTRACT_SHA256,bank_sha256=sha256(ROOT/"pilot_bank.npz"),code_sha256={p:sha256(ROOT/p) for p in ["fit_pilot.py","pilot_common.py"]},scope="development-only; no protected final evaluation",versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),settings=settings,fit_rows=len(fit),assessment_rows=len(assessment),fit_sat_fraction=float(yfit.mean()),assessment_sat_fraction=float(y.mean()),models=metrics,paired_origin_differences=differences,exact_count_oracle=dict(errors=0,total_rows=len(assessment),reason="32-valuation violated-clause count update; independently checked in generation"),elapsed_seconds=time.monotonic()-started)
    np.savez_compressed(ROOT/"predictive_outputs.npz",report_names=np.array(REPORT_NAMES),fit_rows=fit,assessment_rows=assessment,assessment_sat=y,sat_probabilities=np.array(probabilities),selected_new_clauses=np.array(selected),selected_unsat_probabilities=np.array(selection_p),reachable=reachable,per_origin_errors=errors,per_origin_success=successes)
    (ROOT/"predictive_metrics.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(status="PASS",elapsed_seconds=receipt["elapsed_seconds"],paired= differences)),flush=True)


if __name__=="__main__": main()
