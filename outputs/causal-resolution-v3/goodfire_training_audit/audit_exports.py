#!/usr/bin/env python3
"""Retrospective arithmetic on frozen public JSON; standard library, no model."""
import argparse
from collections import Counter
import csv
from decimal import Decimal, ROUND_HALF_UP, getcontext
import hashlib
import json
from pathlib import Path
import re

getcontext().prec = 80
HERE = Path(__file__).resolve().parent
MISSING = object()
CAR = (407, 436, 468, 511, 609, 627, 656, 661, 751, 817)
FIELDS = ("contribution_scale", "specialist_activation_change_sd_ambulance",
          "specialist_activation_change_sd_othercars", "specialist_auc_after",
          "ambulance_margin_drop_ambulance", "ambulance_margin_drop_othercars", "rms_all_logits_change")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(), parse_float=Decimal, parse_constant=Decimal)

def clean(value):
    if isinstance(value, Decimal):
        return str(value)
    if value is MISSING:
        return {"status": "missing"}
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value

def status(value):
    if value is MISSING:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        return "nonnumeric"
    return "finite" if Decimal(value).is_finite() else "nonfinite"

def comparison(candidate, controls):
    support = Counter(status(v) for v in controls)
    valid = [Decimal(v) for v in controls if status(v) == "finite"]
    result = {"candidate": candidate, "candidate_status": status(candidate),
              "control_values_in_source_order": controls, "declared_controls": 16,
              "observed_controls": len(controls), "support": {k:support[k] for k in ("finite", "missing", "null", "nonfinite", "nonnumeric")},
              "control_min": min(valid) if valid else None, "control_max": max(valid) if valid else None,
              "control_sign_counts": {"negative": sum(v<0 for v in valid), "zero": sum(v==0 for v in valid), "positive": sum(v>0 for v in valid)}}
    if status(candidate) == "finite" and valid:
        value = Decimal(candidate)
        less, equal, greater = (sum(v<value for v in valid), sum(v==value for v in valid), sum(v>value for v in valid))
        result.update({"candidate_sign": "negative" if value<0 else "positive" if value>0 else "zero",
                       "controls_lower": less, "controls_equal": equal, "controls_higher": greater,
                       "ascending_rank_interval_including_candidate": [less+1, less+equal+1],
                       "descending_rank_interval_including_candidate": [greater+1, greater+equal+1],
                       "candidate_in_closed_control_range": min(valid)<=value<=max(valid),
                       "rank_denominator": len(valid)+1})
    else:
        result["candidate_in_closed_control_range"] = None
    result["rank_interpretation"] = "Descriptive ranks of exported values only; no randomization or significance interpretation"
    return result

def onset(rows, target, threshold, strict=False):
    def above(row):
        value = row.get(target, MISSING)
        return status(value) == "finite" and (value > threshold if strict else value >= threshold)
    first = next((i for i in range(len(rows)-1) if above(rows[i]) and above(rows[i+1])), None)
    def point(i):
        return None if i is None or i<0 else {k:rows[i].get(k, MISSING) for k in ("step", "epoch", target)}
    return {"threshold": threshold, "operator": ">" if strict else ">=", "row_count":len(rows),
            "first_pair_start_index": first, "previous_sample":point(None if first is None else first-1),
            "first_high_sample": point(first), "confirming_sample": point(None if first is None else first+1),
            "equal_to_threshold_rows": [i for i,r in enumerate(rows) if status(r.get(target,MISSING))=="finite" and r[target]==threshold],
            "undefined_rows": [{"index":i,"status":status(r.get(target,MISSING))} for i,r in enumerate(rows) if status(r.get(target,MISSING))!="finite"],
            "interpretation": "First exported adjacent high-score pair; preceding-sample bracket is not a confidence interval or inferred continuous emergence time"}

def coordinate(value):
    match = re.fullmatch(r"(mean\+|mean-|max\+|min-):stage(\d+)\.block(\d+)\.(\d+)", value)
    if not match:
        return None
    return {"mode":match[1], "stage":int(match[2]), "block":int(match[3]), "unit":int(match[4])}

def numeric_absences(value, path=""):
    rows=[]
    if value is None or isinstance(value, Decimal) and not value.is_finite():
        rows.append({"path":path,"status": "null" if value is None else "nonfinite", "value":value})
    elif isinstance(value,dict):
        for key,item in value.items():
            rows.extend(numeric_absences(item,path+"/"+str(key)))
    elif isinstance(value,list):
        for i,item in enumerate(value):
            rows.extend(numeric_absences(item,path+"/"+str(i)))
    return rows

def build():
    checks, warnings, flat, summary_rows, curve_rows = [], [], [], [], []
    def check(name, condition, details=None):
        checks.append({"name":name, "passed":bool(condition), "details":details})
    freeze=load(HERE/"input_freeze.json")
    for row in freeze["files"]:
        path=HERE/row["path"]
        check("frozen input "+row["path"], path.exists() and sha(path)==row["sha256"] and path.stat().st_size==row["bytes"])
    manifest=load(HERE/"inputs/manifest.json")
    ids, labels=manifest["imageIds"],manifest["imageLabels"]
    label_map=dict(zip(ids,labels))
    counts=Counter(labels)
    check("manifest evaluation ID/label lengths",len(ids)==len(labels)==4210)
    check("manifest evaluation IDs unique",len(set(ids))==4210)
    check("ten car classes each 25",all(counts[c]==25 for c in CAR))
    check("other classes each four",all(n==4 for c,n in counts.items() if c not in CAR) and len([c for c in counts if c not in CAR])==990)
    report= (HERE/"inputs/RESULTS.md").read_text()
    table_rows={}
    for line in report.splitlines():
        if line.startswith(("| Tiny,", "| Wider,")):
            cells=[s.strip() for s in line.strip("|").split("|")]
            table_rows.setdefault(cells[0],[]).append(cells)
    table_names={"fresh-tiny-seed0":"Tiny, seed 0", "fresh-tiny-seed1":"Tiny, seed 1", "fresh-small-seed0":"Wider, seed 0"}
    runs=[]
    absences=[]
    for run in manifest["runs"]:
        rid=run["id"]
        paths={kind:HERE/"inputs"/rid/(kind+".json") for kind in ("metrics","lineage","causal")}
        if not all(path.exists() for path in paths.values()):
            warnings.append({"run":rid,"status":"incomplete inputs", "missing":[k for k,p in paths.items() if not p.exists()]})
            continue
        metrics,lineage,causal=(load(paths[k]) for k in ("metrics","lineage","causal"))
        for kind,data in (("metrics",metrics),("lineage",lineage),("causal",causal)):
            absences.extend(numeric_absences(data,rid+"/"+kind))
        check(rid+" units/name count",len(run["names"])==run["units"]==sum(b["count"] for b in run["blocks"]))
        check(rid+" sixteen snapshot steps",[r["step"] for r in lineage["rows"]]==[r["step"] for r in metrics["rows"]]==run["steps"] and len(run["steps"])==16)
        check(rid+" epoch alignment",[r["epoch"] for r in lineage["rows"]]==[r["epoch"] for r in metrics["rows"]])
        errors=[abs(r["epoch"]-Decimal(r["step"])/Decimal(2502)) for r in lineage["rows"]]
        check(rid+" represented epoch step/2502",max(errors)<=Decimal("1e-12"),{"max_absolute_decimal_discrepancy":max(errors),"tolerance":"1e-12"})
        onset_results={}
        block_positions={b["id"]:i for i,b in enumerate(run["blocks"])}
        parsed={}
        for target,threshold in (("broad",Decimal("0.85")),("ambulance",Decimal("0.80")),("cab",Decimal("0.80"))):
            definition=lineage["targets"][target]
            c=coordinate(definition["coordinate"])
            parsed[target]=c
            start=run["starts"][target]
            expected_name=manifest["modes"][start["mode"]]+":"+run["names"][start["unit"]]
            check(rid+" "+target+" coordinate and mode index",definition["coordinate"]==start["coordinate"]==expected_name and definition["index"]==start["mode"]*run["units"]+start["unit"])
            final_metrics=metrics["rows"][-1]["broad"] if target=="broad" else metrics["rows"][-1]["specialists"][target]
            check(rid+" "+target+" final metric/lineage definition",definition==final_metrics)
            check(rid+" "+target+" final lineage score",lineage["rows"][-1][target]==definition["test_auc"])
            inclusive=onset(lineage["rows"],target,threshold)
            strict=onset(lineage["rows"],target,threshold,True)
            onset_results[target]={"inclusive":inclusive,"strict_first_pair_start_index":strict["first_pair_start_index"],
                                   "strict_vs_inclusive_changes_first_pair":strict["first_pair_start_index"]!=inclusive["first_pair_start_index"]}
            for i,row in enumerate(lineage["rows"]):
                curve_rows.append({"run":rid,"target":target,"coordinate":definition["coordinate"],"source_row":i,
                                   "step":row["step"],"epoch":row["epoch"],"evaluation_auroc":row.get(target,MISSING),"threshold":threshold})
        broad_pos=block_positions[f"stage{parsed['broad']['stage']}.block{parsed['broad']['block']}"]
        ambulance_pos=block_positions[f"stage{parsed['ambulance']['stage']}.block{parsed['ambulance']['block']}"]
        candidate=causal["candidate"]
        cc=coordinate(candidate["coordinate"])
        candidate_pos=block_positions[f"stage{cc['stage']}.block{cc['block']}"]
        check(rid+" final ambulance upstream of final broad",ambulance_pos<broad_pos)
        check(rid+" precursor strictly upstream of ambulance",candidate_pos<ambulance_pos)
        check(rid+" precursor target matches lineage",candidate["specialist"]==lineage["targets"]["ambulance"]["coordinate"])
        check(rid+" precursor source selection step",candidate["selection_step"]==2500 and candidate["selection_step"] in run["steps"])
        cids,clabels=causal["evaluation_ids"],causal["evaluation_labels"]
        ccounts=Counter(clabels)
        check(rid+" causal cohort lengths and uniqueness",len(cids)==len(clabels)==len(set(cids))==500)
        check(rid+" causal cohort subset/labels",all(label_map.get(i)==label for i,label in zip(cids,clabels)))
        check(rid+" causal cohort car counts",all(ccounts[c]==25 for c in CAR) and sum(n for c,n in ccounts.items() if c not in CAR)==250)
        road_counts=sum(counts[int(c)] for c in metrics["road_classes"])
        check(rid+" hard-road count",len(metrics["road_classes"])==11 and road_counts==44)
        rows=causal["rows"]
        check(rid+" five intervention times",len(rows)==5)
        aligned=[next((r for r in lineage["rows"] if r["epoch"]==row["epoch"]),None) for row in rows]
        check(rid+" intervention step set",[r["step"] if r else None for r in aligned]==[0,2500,5000,12500,25020])
        comparisons=[]
        control_sets=[]
        for ti,row in enumerate(rows):
            ablations=row["ablations"]
            candidates=[a for a in ablations if a.get("is_candidate") is True]
            controls=[a for a in ablations if a.get("is_candidate") is False]
            check(rid+f" time{ti} candidate/control denominator",len(candidates)==1 and len(controls)==16 and len(ablations)==17)
            check(rid+f" time{ti} unique local units",len({a["unit"] for a in ablations})==len(ablations))
            check(rid+f" time{ti} same-block unit bounds",all(0<=a["unit"]<run["blocks"][candidate_pos]["count"] for a in ablations))
            if len(candidates)!=1:
                warnings.append({"run":rid,"time_index":ti,"status":"candidate comparison unavailable"})
                continue
            selected=candidates[0]
            check(rid+f" time{ti} literal candidate unit",selected["unit"]==cc["unit"])
            base=row.get("specialist_auc_before",MISSING)
            check(rid+f" time{ti} baseline equals fixed lineage",aligned[ti] is not None and base==aligned[ti]["ambulance"])
            control_sets.append([a["unit"] for a in controls])
            for ai,a in enumerate(ablations):
                item={"run":rid,"time_index":ti,"source_row_order":ai,"epoch":row["epoch"],
                      "step":aligned[ti]["step"] if aligned[ti] else None,"unit":a["unit"],"is_candidate":a.get("is_candidate",MISSING),
                      "specialist_auc_before":base}
                item.update({k:a.get(k,MISSING) for k in FIELDS})
                item["specialist_auc_decrease"]=base-a["specialist_auc_after"] if status(base)==status(a.get("specialist_auc_after",MISSING))=="finite" else MISSING
                flat.append(item)
            fields={key:comparison(selected.get(key,MISSING),[a.get(key,MISSING) for a in controls]) for key in FIELDS}
            def drop(a):
                return base-a["specialist_auc_after"] if status(base)==status(a.get("specialist_auc_after",MISSING))=="finite" else MISSING
            fields["specialist_auc_decrease"]=comparison(drop(selected),[drop(a) for a in controls])
            for key,comp in fields.items():
                summary_rows.append({"run":rid,"time_index":ti,"epoch":row["epoch"],"field":key,
                    **{k:v for k,v in comp.items() if k not in ("control_values_in_source_order","rank_interpretation","support","control_sign_counts")},
                    **{"n_"+k:v for k,v in comp["support"].items()},**{"control_"+k:v for k,v in comp["control_sign_counts"].items()}})
            comparisons.append({"time_index":ti,"epoch":row["epoch"],"step":aligned[ti]["step"] if aligned[ti] else None,
                                "baseline_auc":base,"candidate_unit":selected["unit"],"control_units_in_source_order":[a["unit"] for a in controls],"fields":fields})
        source_rows=table_rows[table_names[rid]]
        onset_table=next(c for c in source_rows if len(c)==6)
        causal_table=next(c for c in source_rows if len(c)==4)
        report_checks=[]
        def reported_numeric(label,text,actual):
            value=Decimal(text.replace("−","-").replace("+",""))
            unit=Decimal(1).scaleb(value.as_tuple().exponent)
            ok=abs(Decimal(actual)-value)<=unit/2
            item={"field":label,"report_string":text,"reported_decimal":value,"exported_or_recalculated_value":actual,"half_last_decimal_unit":unit/2,"agrees_at_report_precision":ok}
            report_checks.append(item)
            check(rid+" report "+label,ok,item)
        for i,target in ((1,"broad"),(2,"ambulance")):
            boundaries=[int(x) for x in re.findall(r"\d+",onset_table[i])]
            found=onset_results[target]["inclusive"]
            points=[found["previous_sample"],found["first_high_sample"]]
            recalculated=[int(p["epoch"].quantize(Decimal(1),rounding=ROUND_HALF_UP)) for p in points] if all(p is not None for p in points) else None
            check(rid+" report "+target+" onset bracket",boundaries==recalculated,{"source_text":onset_table[i],"rounded_sample_epochs":recalculated})
        reported_numeric("final broad AUROC",onset_table[3],lineage["targets"]["broad"]["test_auc"])
        reported_numeric("final hard-road AUROC",onset_table[4],lineage["targets"]["broad"]["road_vehicle_auc"])
        reported_numeric("final ambulance AUROC",onset_table[5],lineage["targets"]["ambulance"]["test_auc"])
        before_after=[x.strip() for x in causal_table[2].split("→")]
        final=comparisons[-1]
        reported_numeric("final causal AUROC before",before_after[0],final["baseline_auc"])
        reported_numeric("final causal AUROC after",before_after[1],final["fields"]["specialist_auc_after"]["candidate"])
        reported_numeric("final response decrease SD",causal_table[3],final["fields"]["specialist_activation_change_sd_ambulance"]["candidate"])
        named=re.search(r"S(\d+)\.B(\d+)\.N(\d+), (negative|positive) mean",causal_table[1])
        check(rid+" report candidate name",named is not None and [cc[k] for k in ("stage","block","unit")]==[int(named[i]) for i in (1,2,3)] and cc["mode"]==("mean-" if named[4]=="negative" else "mean+"))
        check(rid+" report no sustained cab",onset_results["cab"]["inclusive"]["first_pair_start_index"] is None)
        for field in ("specialist_activation_change_sd_ambulance","specialist_auc_after"):
            check(rid+" report final candidate in controls "+field, final["fields"][field]["candidate_in_closed_control_range"] is True)
        runs.append({"run_id":rid,"title":run["title"],"parameters_source_reported":run["parameters"],"native_units":run["units"],
                     "lineage_selection_source_statement":lineage["selection"],"candidate_selection_source_statement":candidate,
                     "control_matching_source_statement":causal["controls"],"targets":lineage["targets"],"onsets":onset_results,
                     "fixed_curves":lineage["rows"],"initial_readout_auc_source_reported":metrics["rows"][0]["readout_auc"],
                     "causal_cohort_counts":dict(sorted(ccounts.items())),"causal_cohort_id_count":len(cids),
                     "causal_cohort_ambulance":ccounts[407],"causal_cohort_other_car":sum(ccounts[c] for c in CAR if c!=407),
                     "hard_road_negative_count":road_counts,"comparisons":comparisons,
                     "control_units_by_time":control_sets,"same_control_set_at_every_time":all(set(s)==set(control_sets[0]) for s in control_sets),
                     "source_report_rows":source_rows,"numeric_report_reconciliation":report_checks})
    readouts=[r["initial_readout_auc_source_reported"] for r in runs]
    check("report initial readout range lower rounded",abs(min(readouts)-Decimal("0.855"))<=Decimal("0.0005"))
    check("report initial readout range upper rounded",abs(max(readouts)-Decimal("0.874"))<=Decimal("0.0005"))
    check("all three manifest runs audited",len(runs)==len(manifest["runs"])==3)
    check("all 255 exported ablations retained",len(flat)==255)
    check("all 120 field comparisons retained",len(summary_rows)==120)
    check("all 144 fixed-trajectory score rows retained",len(curve_rows)==144)
    result={"scope":"Retrospective arithmetic/provenance audit of written source exports; no experimental replication, raw-response AUROC estimation, fitting or neural inference",
            "decimal_policy":"Numbers are represented as decimal strings exactly as parsed; arithmetic uses 80 significant decimal digits. Ties are at exported precision.",
            "input_freeze_sha256":sha(HERE/"input_freeze.json"),"audit_code_sha256":sha(Path(__file__)),
            "source_manifest_statements":{k:manifest[k] for k in ("sample","sampling","identity","selection","training","associations","scores")},
            "manifest_class_counts":dict(sorted(counts.items())),"runs":runs,"undefined_source_cells":absences,"warnings":warnings,
            "counts":{"runs":len(runs),"fixed_curve_rows":len(curve_rows),"ablation_rows":len(flat),"candidate_control_field_comparisons":len(summary_rows),"checks":len(checks)},
            "checks":checks,"passed":all(c["passed"] for c in checks)}
    return result,flat,summary_rows,curve_rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true",help="Recalculate in memory and compare with saved audit.json; no files written.")
    args=parser.parse_args()
    result,flat,comparisons,curves=build()
    output=clean(result)
    if args.verify:
        expected=json.loads((HERE/"audit.json").read_text())
        equal=expected==output
        print(json.dumps({"matches_saved_audit":equal,"internal_checks_pass":result["passed"],"counts":result["counts"],"audit_sha256":sha(HERE/"audit.json")}))
        if not equal:
            raise SystemExit(1)
    else:
        with (HERE/"audit.json").open("x") as stream:
            json.dump(output,stream,indent=2,ensure_ascii=False)
            stream.write("\n")
        for name,rows in (("ablations.csv",flat),("candidate_control_comparisons.csv",comparisons),("fixed_coordinate_curves.csv",curves)):
            with (HERE/name).open("x",newline="") as stream:
                columns=list(dict.fromkeys(k for row in rows for k in row))
                writer=csv.DictWriter(stream,fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list)) or v is MISSING else clean(v) for k,v in row.items()})
        print(json.dumps({"passed":result["passed"],"counts":result["counts"],"failed_checks":[c for c in output["checks"] if not c["passed"]]},indent=2))

if __name__=="__main__":
    main()
