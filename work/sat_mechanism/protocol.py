"""Shared frozen protocol. Importing this module performs no neural inference."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / "sat_case"
SOURCES = ROOT.parent / "next_phase"
sys.path.insert(0, str(PILOT))
from pilot_common import CLAUSES, CLAUSE_TRUTH, PUBLISHED_UNITS, encode

CONTRACT_SHA = "9f25cb332a06c10dacb1315c8b5f7b7f7bbb9ef8a082c3cd2802f219902d3176"
MAPPING_SHA = "5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d"
BANK_SHA = "610d02cc85ba825b7ac3100efd30ad1ec8f90f02e11a9fe8444f0c62a9e2971b"
UNITS = [10,41,55,77,81,185,224,245,261,291,317,326,380,411,416,492,495]
MODELS = ["structured", "post_score", "mismatched_unit", "mismatched_assignment", "context_only", "symbolic_oracle"]
CODE_FILES = ["protocol.py", "prepare_freeze.py", "native.py", "analyze.py", "selfcheck.py", "run_study.py"]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(name): return json.loads((ROOT/name).read_text())


def mapping():
    data = read_json("source_mapping.json")
    rows = data["single_assignment_general_units"]
    assert [r["unit"] for r in rows] == UNITS
    assert len(set(r["assignment"] for r in rows)) == 16
    for i, r in enumerate(rows):
        assert r["truth_table_index"] == int(r["assignment"].replace("T","1").replace("F","0"),2)
        assert r["assignment"] != rows[(i+1)%17]["assignment"]
    return rows


def frozen():
    receipt = read_json("freeze.json")
    for path, expected in receipt["sha256"].items():
        if sha(ROOT/path) != expected: raise ValueError("Frozen input changed: " + path)
    verify_pilot()
    return receipt


def verify_pilot():
    manifest=json.loads((PILOT/"MANIFEST.json").read_text())
    for name, entry in manifest["files"].items():
        if sha(PILOT/name)!=entry["sha256"]: raise ValueError("Pilot changed: "+name)
    assert sha(PILOT/"MANIFEST.json")=="b81f234d4c20185955760ae01ffdfea3239b4c6c1d204666792819b315524e82"


def support_metrics(y, pred, weights=None):
    y=np.asarray(y,dtype=bool);pred=np.asarray(pred)
    weights=np.ones(len(y),dtype=np.int64) if weights is None else np.asarray(weights)
    support=int(weights.sum());positive=int(weights[y].sum());negative=int(weights[~y].sum())
    if not np.isfinite(pred).all():
        return dict(n=support,positive=positive,negative=negative,tp=None,tn=None,fp=None,fn=None,sensitivity=None,specificity=None,balanced_accuracy=None,reason="unestimable predictor")
    pred=pred.astype(bool)
    tp=int(weights[y&pred].sum());tn=int(weights[(~y)&(~pred)].sum())
    fp=negative-tn;fn=positive-tp
    sensitivity=tp/positive if positive else None;specificity=tn/negative if negative else None
    return dict(n=support,positive=positive,negative=negative,tp=tp,tn=tn,fp=fp,fn=fn,sensitivity=sensitivity,specificity=specificity,balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,reason=None if positive and negative else "missing outcome class")


def strict_mean(values):
    return float(sum(values)/len(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None


def assignment_macro(values, rows=None):
    rows=mapping() if rows is None else rows
    assignments=sorted(set(r["assignment"] for r in rows))
    by={a:strict_mean([values[i] for i,r in enumerate(rows) if r["assignment"]==a]) for a in assignments}
    return dict(assignments=by,macro=strict_mean(list(by.values())))


def apply_rule(score, rule):
    if rule["kind"]=="constant": return np.full(np.shape(score),rule["value"],dtype=bool)
    threshold=np.frombuffer(bytes.fromhex(rule["threshold_float32_le_hex"]),dtype="<f4")[0]
    return np.asarray(score)>threshold


def threshold_rule(score, y):
    """Exact integer objective; strict >; constant rules represented explicitly."""
    score=np.asarray(score,dtype=np.float32);y=np.asarray(y,dtype=bool)
    assert np.isfinite(score).all()
    n1=int(y.sum());n0=len(y)-n1
    def constant(value, estimable):
        tau=np.float32(-np.inf if value else np.inf)
        return dict(kind="constant",value=bool(value),comparison="constant",threshold=None,threshold_float32_le_hex=tau.astype("<f4").tobytes().hex(),fit_estimable=estimable,positive=n1,negative=n0,predicted_positive=len(y) if value else 0,balanced_error=0.5 if estimable else None,candidate_count=int(len(np.unique(score))+2),balanced_error_integer_numerator=n1*n0 if estimable else None,balanced_error_integer_denominator=2*n1*n0 if estimable else None)
    if not n1 or not n0:return constant(False,False)
    unique,inverse=np.unique(score,return_inverse=True)
    pos=np.bincount(inverse,weights=y,minlength=len(unique)).astype(np.int64)
    neg=np.bincount(inverse,weights=~y,minlength=len(unique)).astype(np.int64)
    fn=np.cumsum(pos);fp=n0-np.cumsum(neg)
    errors=fn*n0+fp*n1
    predictions=n1-fn+fp
    best=(n1*n0,0,-float("inf"))
    winner=constant(False,True)
    for j,tau in enumerate(unique):
        key=(int(errors[j]),int(predictions[j]),-float(tau))
        if key<best:
            best=key
            winner=dict(kind="threshold",value=None,comparison=">",threshold=float(tau),threshold_float32_le_hex=np.asarray(tau,dtype="<f4").tobytes().hex(),fit_estimable=True,positive=n1,negative=n0,predicted_positive=int(predictions[j]),balanced_error=int(errors[j])/(2*n1*n0),candidate_count=len(unique)+2,balanced_error_integer_numerator=int(errors[j]),balanced_error_integer_denominator=2*n1*n0)
    # Constant one has equal balanced error to constant zero, more positives, so never wins.
    return winner


def context_design(rows):
    rows=np.asarray(rows,dtype=np.int64)
    out=np.zeros((len(rows),90),dtype=np.float64)
    k=np.arange(len(rows));out[k,rows[:,0]]=1;out[k,10+rows[:,1]]=1;out[k,50+rows[:,2]]=1
    return out


def percentile_interval(values):
    a=np.asarray(values,dtype=float);valid=a[np.isfinite(a)]
    return dict(draws=len(a),valid_draws=len(valid),na_draws=int(len(a)-len(valid)),interval_2_5_97_5=np.quantile(valid,[.025,.975]).tolist() if len(valid)>=1900 else None)
