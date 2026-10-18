#!/usr/bin/env python3
"""Separate exact-rational replay of exported onset and comparison arithmetic."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text(),parse_float=str)
def f(value):
    return Fraction(str(value))

audit=read(HERE/"audit.json")
checks=[]
for run in audit["runs"]:
    rid=run["run_id"]
    lineage=read(HERE/"inputs"/rid/"lineage.json")
    causal=read(HERE/"inputs"/rid/"causal.json")
    for target,threshold in (("broad",Fraction(17,20)),("ambulance",Fraction(4,5)),("cab",Fraction(4,5))):
        values=[f(row[target]) for row in lineage["rows"]]
        starts=[i for i,(a,b) in enumerate(zip(values,values[1:])) if a>=threshold and b>=threshold]
        actual=run["onsets"][target]["inclusive"]
        checks.append(actual["first_pair_start_index"]==(starts[0] if starts else None))
        checks.append(actual["equal_to_threshold_rows"]==[i for i,v in enumerate(values) if v==threshold])
    for row,comparison in zip(causal["rows"],run["comparisons"]):
        candidate=[x for x in row["ablations"] if x["is_candidate"]]
        controls=[x for x in row["ablations"] if not x["is_candidate"]]
        checks.append(len(candidate)==1 and len(controls)==16)
        checks.append(comparison["control_units_in_source_order"]==[x["unit"] for x in controls])
        for field,saved in comparison["fields"].items():
            def val(record):
                if field=="specialist_auc_decrease":
                    return f(row["specialist_auc_before"])-f(record["specialist_auc_after"])
                return f(record[field])
            x=val(candidate[0])
            values=[val(c) for c in controls]
            lower=sum(y<x for y in values)
            equal=sum(y==x for y in values)
            greater=sum(y>x for y in values)
            checks.extend([f(saved["candidate"])==x,
                           [f(v) for v in saved["control_values_in_source_order"]]==values,
                           f(saved["control_min"])==min(values),f(saved["control_max"])==max(values),
                           saved["controls_lower"]==lower,saved["controls_equal"]==equal,saved["controls_higher"]==greater,
                           saved["ascending_rank_interval_including_candidate"]==[1+lower,1+lower+equal],
                           saved["descending_rank_interval_including_candidate"]==[1+greater,1+greater+equal],
                           saved["candidate_in_closed_control_range"]==(min(values)<=x<=max(values)),
                           saved["control_sign_counts"]=={"negative":sum(v<0 for v in values),"zero":sum(v==0 for v in values),"positive":sum(v>0 for v in values)}])
result={"passed":all(checks),"mechanical_checks":len(checks),"audit_sha256":hashlib.sha256((HERE/"audit.json").read_bytes()).hexdigest(),
        "verifier_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arithmetic":"Fraction from the exported decimal strings; separate implementation from the Decimal audit",
        "independence_boundary":"Separate arithmetic code by the same audit agent, not independent investigators or experimental replication",
        "scope":"Nine onset sequences and all 120 signed field comparisons, including all16 controls; no raw-response AUROC estimation or neural inference"}
print(json.dumps(result,indent=2))
if not result["passed"]:
    raise SystemExit(1)
