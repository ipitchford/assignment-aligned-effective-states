"""Independent, standard-library arithmetic on frozen Goodfire measurement exports.

Never imports or executes the audit's programs. No network, model, fit, bootstrap,
or reconstruction of AUROC from responses. --numerical-only permits an explicitly
limited redistributed layout without the original copyrighted prose.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import re
import time

MANIFEST_SHA = "2bde68b01eff9c0b4b9eb799b82a59cad352d11a6ab9e414cabb05479f8a6cd4"
CAR = [407,436,468,511,609,627,656,661,751,817]
RUNS = ["fresh-tiny-seed0","fresh-tiny-seed1","fresh-small-seed0"]
FIELDS = ["contribution_scale","specialist_activation_change_sd_ambulance","specialist_activation_change_sd_othercars",
          "specialist_auc_after","ambulance_margin_drop_ambulance","ambulance_margin_drop_othercars","rms_all_logits_change","specialist_auc_decrease"]
THRESHOLDS = {"broad":Q(17,20),"ambulance":Q(4,5),"cab":Q(4,5)}
NUMERICAL_REQUIRED = ["manifest.json","input_freeze.json","audit.json","inputs/manifest.json",
                      "ablations.csv","candidate_control_comparisons.csv","fixed_coordinate_curves.csv"] + ["inputs/"+r+"/"+k+".json" for r in RUNS for k in ["metrics","lineage","causal"]]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def number(value):
    if isinstance(value,bool) or value is None: raise ValueError("Boolean/null is not a measurement")
    return value if isinstance(value,Q) else Q(str(value))
def decimal(value):
    with localcontext() as context:
        context.prec=90
        return str(Decimal(value.numerator)/Decimal(value.denominator))
def clean(value):
    if isinstance(value,Q): return decimal(value)
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    return value


class Audit:
    def __init__(self,base,output,numerical_only):
        self.base,self.output,self.numeric=base,output,numerical_only
        self.counts=Counter(); self.failures=[]; self.identities={}; self.omitted=[]
    def check(self,category,condition,location=""):
        self.counts[category]+=1
        if not condition: self.failures.append({"category":category,"location":location})
    def load(self,relative):
        p=self.base/relative
        self.identities[relative]={"sha256":sha(p),"bytes":p.stat().st_size}
        def reject_constant(x): raise ValueError("Nonfinite JSON constant "+x+" in "+relative)
        return json.loads(p.read_text(),parse_float=Q,parse_constant=reject_constant)
    def csv(self,name):
        p=self.base/name;self.identities[name]={"sha256":sha(p),"bytes":p.stat().st_size}
        with p.open(newline="") as stream:return list(csv.DictReader(stream))
    def equal_number(self,actual,expected,category,location):self.check(category,number(actual)==number(expected),location)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--audit-dir",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--numerical-only",action="store_true")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();a=Audit(args.audit_dir.resolve(),args.output.resolve(),args.numerical_only)
    results={"onsets":[],"comparisons":[],"exceptions":[],"controls":[],"run_summaries":[]}
    error=None
    try:
        a.check("root_manifest_identity",sha(a.base/"manifest.json")==MANIFEST_SHA)
        manifest=a.load("manifest.json"); freeze=a.load("input_freeze.json")
        bound={}
        for source in [manifest,freeze]:
            for r in source["files"]:
                if r["path"] in bound:a.check("consistent_nested_binding",bound[r["path"]]==r,r["path"])
                bound[r["path"]]=r
        for relative,r in bound.items():
            p=a.base/relative
            if not p.exists() and a.numeric and relative not in NUMERICAL_REQUIRED:
                a.omitted.append(relative);continue
            a.check("bound_file_present",p.is_file(),relative)
            if not p.is_file():raise FileNotFoundError(p)
            actual={"sha256":sha(p),"bytes":p.stat().st_size};a.identities[relative]=actual
            a.check("bound_file_hash",actual["sha256"]==r["sha256"],relative)
            a.check("bound_file_size",actual["bytes"]==r["bytes"],relative)
        for relative in NUMERICAL_REQUIRED:a.check("required_numerical_input",(a.base/relative).is_file(),relative)
        source=a.load("inputs/manifest.json"); saved=a.load("audit.json")
        a.check("audit_input_binding",saved["input_freeze_sha256"]==sha(a.base/"input_freeze.json"))
        a.check("audit_code_binding",saved["audit_code_sha256"]==bound["audit_exports.py"]["sha256"])
        a.check("audit_check_receipt",saved["counts"]["checks"]==len(saved["checks"])==206 and all(c["passed"] for c in saved["checks"]))
        a.check("reported_audit_state",saved["passed"] is True and saved["warnings"]==[] and saved["undefined_source_cells"]==[])
        for key in ["sample","sampling","identity","selection","training","associations","scores"]:
            a.check("source_statement_preserved",source[key]==saved["source_manifest_statements"][key],key)
        a.check("run_inventory",[r["id"] for r in source["runs"]]==RUNS==[r["run_id"] for r in saved["runs"]])
        images,labels=source["imageIds"],source["imageLabels"]
        a.check("evaluation_cohort",len(images)==len(labels)==len(set(images))==4210)
        label_lookup=dict(zip(images,labels));counts=Counter(labels)
        a.check("evaluation_class_count",len(counts)==1000)
        for c in range(1000):a.check("evaluation_class_support",counts[c]==(25 if c in CAR else 4),str(c))
        a.check("saved_class_counts",saved["manifest_class_counts"]=={str(k):v for k,v in sorted(counts.items())})
        curves=a.csv("fixed_coordinate_curves.csv");ablations=a.csv("ablations.csv");comparisons=a.csv("candidate_control_comparisons.csv")
        a.check("curve_row_total",len(curves)==144);a.check("ablation_row_total",len(ablations)==255);a.check("comparison_row_total",len(comparisons)==120)
        curve_idx={(r["run"],r["target"],int(r["source_row"])):r for r in curves}
        ablation_idx={(r["run"],int(r["time_index"]),int(r["source_row_order"])):r for r in ablations}
        comparison_idx={(r["run"],int(r["time_index"]),r["field"]):r for r in comparisons}
        a.check("curve_row_keys",len(curve_idx)==144);a.check("ablation_row_keys",len(ablation_idx)==255);a.check("comparison_row_keys",len(comparison_idx)==120)
        def check_source_tree(obj,path,range_context=False):
            if isinstance(obj,dict):
                for key,value in obj.items():check_source_tree(value,path+"/"+key,range_context or "auc" in key or "ci95" in key)
            elif isinstance(obj,list):
                for i,value in enumerate(obj):check_source_tree(value,path+"/"+str(i),range_context)
            else:
                a.check("source_nonnull_leaf",obj is not None,path)
                if isinstance(obj,(int,Q)) and not isinstance(obj,bool) and range_context:a.check("source_auc_or_ci_range",0<=obj<=1,path)
        for metadata,reported in zip(source["runs"],saved["runs"]):
            rid=metadata["id"]
            data={k:a.load("inputs/"+rid+"/"+k+".json") for k in ["metrics","lineage","causal"]}
            metrics,lineage,causal=data["metrics"],data["lineage"],data["causal"]
            for kind,obj in data.items():check_source_tree(obj,rid+"/"+kind)
            blocks=metadata["blocks"];offset=0;block_positions={}
            for bi,block in enumerate(blocks):
                a.check("block_offsets",block["start"]==offset,rid+"/"+block["id"]);block_positions[block["id"]]=bi
                for u in range(block["count"]):a.check("literal_native_name",metadata["names"][offset+u]==block["id"]+"."+str(u),rid+"/"+str(offset+u))
                offset+=block["count"]
            a.check("native_coordinate_count",len(metadata["names"])==metadata["units"]==offset,rid)
            a.check("snapshot_count",len(lineage["rows"])==len(metrics["rows"])==len(metadata["steps"])==16,rid)
            a.check("step_order",metadata["steps"]==sorted(set(metadata["steps"])),rid)
            for row,mrow,step in zip(lineage["rows"],metrics["rows"],metadata["steps"]):
                a.check("snapshot_alignment",row["step"]==mrow["step"]==step and row["epoch"]==mrow["epoch"],rid+"/"+str(step))
                a.check("epoch_step_ratio",abs(row["epoch"]-Q(step,2502))<=Q(1,10**12),rid+"/"+str(step))
            coord_blocks={}
            for target,threshold in THRESHOLDS.items():
                definition=lineage["targets"][target];startdef=metadata["starts"][target]
                a.check("fixed_coordinate_decoding",definition["coordinate"]==startdef["coordinate"]==source["modes"][startdef["mode"]]+":"+metadata["names"][startdef["unit"]],rid+"/"+target)
                a.check("fixed_coordinate_index",definition["index"]==startdef["mode"]*metadata["units"]+startdef["unit"],rid+"/"+target)
                coord_blocks[target]=definition["coordinate"].split(":")[1].rsplit(".",1)[0]
                final=metrics["rows"][-1]["broad"] if target=="broad" else metrics["rows"][-1]["specialists"][target]
                a.check("final_selected_definition",final==definition,rid+"/"+target)
                a.equal_number(lineage["rows"][-1][target],definition["test_auc"],"final_lineage_score",rid+"/"+target)
                values=[number(row[target]) for row in lineage["rows"]]
                inclusive=[i for i in range(15) if min(values[i:i+2])>=threshold]
                strict=[i for i in range(15) if min(values[i:i+2])>threshold]
                index=inclusive[0] if inclusive else None
                onset=reported["onsets"][target]["inclusive"]
                a.check("onset_index",onset["first_pair_start_index"]==index,rid+"/"+target)
                a.equal_number(onset["threshold"],threshold,"onset_threshold",rid+"/"+target)
                a.check("onset_equality_and_missing",onset["equal_to_threshold_rows"]==[i for i,v in enumerate(values) if v==threshold] and onset["undefined_rows"]==[],rid+"/"+target)
                a.check("strict_onset",reported["onsets"][target]["strict_first_pair_start_index"]==(strict[0] if strict else None) and not reported["onsets"][target]["strict_vs_inclusive_changes_first_pair"],rid+"/"+target)
                for key,si in [("previous_sample",None if index is None else index-1),("first_high_sample",index),("confirming_sample",None if index is None else index+1)]:
                    if si is None or si<0:a.check("onset_boundary",onset[key] is None,rid+"/"+target+"/"+key)
                    else:
                        for field in ["step","epoch",target]:a.equal_number(onset[key][field],lineage["rows"][si][field],"onset_boundary",rid+"/"+target+"/"+key+"/"+field)
                results["onsets"].append({"run":rid,"target":target,"first_pair_start_index":index,"threshold":threshold,
                    "previous_step":None if index is None or index==0 else lineage["rows"][index-1]["step"],
                    "first_step":None if index is None else lineage["rows"][index]["step"],"confirm_step":None if index is None else lineage["rows"][index+1]["step"]})
                for si,row in enumerate(lineage["rows"]):
                    savedrow=curve_idx[(rid,target,si)]
                    a.check("curve_coordinate",savedrow["coordinate"]==definition["coordinate"],rid+"/"+target+"/"+str(si))
                    for field,value in [("step",row["step"]),("epoch",row["epoch"]),("evaluation_auroc",row[target]),("threshold",threshold)]:a.equal_number(savedrow[field],value,"curve_csv_value",rid+"/"+target+"/"+str(si)+"/"+field)
                    for field in ["step","epoch",target]:a.equal_number(reported["fixed_curves"][si][field],row[field],"audit_curve_value",rid+"/"+target+"/"+str(si)+"/"+field)
            candidate=causal["candidate"]; candidate_name=candidate["coordinate"].split(":")[1];candidate_block,candidate_unit=candidate_name.rsplit(".",1)
            a.check("causal_block_order",block_positions[candidate_block]<block_positions[coord_blocks["ambulance"]]<block_positions[coord_blocks["broad"]],rid)
            a.check("candidate_fixed_identity",candidate["specialist"]==lineage["targets"]["ambulance"]["coordinate"] and candidate["selection_step"]==2500,rid)
            cids,clabels=causal["evaluation_ids"],causal["evaluation_labels"]
            a.check("intervention_cohort_size",len(cids)==len(clabels)==len(set(cids))==500,rid)
            for ident,label in zip(cids,clabels):a.check("intervention_id_label_subset",label_lookup.get(ident)==label,rid+"/"+str(ident))
            cc=Counter(clabels)
            for cls in CAR:a.check("intervention_car_support",cc[cls]==25,rid+"/"+str(cls))
            a.check("intervention_noncar_support",sum(v for k,v in cc.items() if k not in CAR)==250,rid)
            a.check("hard_road_support",len(metrics["road_classes"])==11 and sum(counts[int(k)] for k in metrics["road_classes"])==44,rid)
            a.check("intervention_time_count",len(causal["rows"])==len(reported["comparisons"])==5,rid)
            control_sets=[]
            for ti,(row,summary) in enumerate(zip(causal["rows"],reported["comparisons"])):
                anchor=[x for x in lineage["rows"] if x["epoch"]==row["epoch"]]
                a.check("intervention_snapshot",len(anchor)==1 and anchor[0]["step"]==[0,2500,5000,12500,25020][ti],rid+"/"+str(ti))
                step=anchor[0]["step"];baseline=number(row["specialist_auc_before"])
                a.equal_number(anchor[0]["ambulance"],baseline,"intervention_fixed_baseline",rid+"/"+str(ti))
                candidates=[r for r in row["ablations"] if r["is_candidate"] is True]; controls=[r for r in row["ablations"] if r["is_candidate"] is False]
                a.check("candidate_control_support",len(candidates)==1 and len(controls)==16 and len(row["ablations"])==17,rid+"/"+str(ti))
                a.check("unique_same_block_units",len({r["unit"] for r in row["ablations"]})==17 and all(0<=r["unit"]<blocks[block_positions[candidate_block]]["count"] for r in row["ablations"]),rid+"/"+str(ti))
                a.check("candidate_unit",candidates[0]["unit"]==int(candidate_unit),rid+"/"+str(ti))
                control_sets.append([r["unit"] for r in controls])
                a.check("control_order",summary["control_units_in_source_order"]==control_sets[-1]==reported["control_units_by_time"][ti],rid+"/"+str(ti))
                def value(record,field):return baseline-number(record["specialist_auc_after"]) if field=="specialist_auc_decrease" else number(record[field])
                for ai,record in enumerate(row["ablations"]):
                    csvrow=ablation_idx[(rid,ti,ai)]
                    a.check("ablation_row_identity",int(csvrow["unit"])==record["unit"] and csvrow["is_candidate"]==str(record["is_candidate"]),rid+"/"+str(ti)+"/"+str(ai))
                    for field,number0 in [("epoch",row["epoch"]),("step",step),("specialist_auc_before",baseline)]+[(f,value(record,f)) for f in FIELDS]:a.equal_number(csvrow[field],number0,"ablation_csv_value",rid+"/"+str(ti)+"/"+str(ai)+"/"+field)
                for field in FIELDS:
                    x=value(candidates[0],field); vv=[value(r,field) for r in controls]; ordered=sorted(vv)
                    low=sum(v<x for v in vv);equal=vv.count(x);high=16-low-equal
                    within=ordered[0]<=x<=ordered[-1]
                    calculated={"candidate":x,"control_min":ordered[0],"control_max":ordered[-1],"controls_lower":low,"controls_equal":equal,"controls_higher":high,
                        "ascending_rank_interval_including_candidate":[low+1,low+equal+1],"descending_rank_interval_including_candidate":[high+1,high+equal+1],
                        "candidate_in_closed_control_range":within,"rank_denominator":17,"candidate_sign":"negative" if x<0 else "positive" if x>0 else "zero"}
                    existing=summary["fields"][field]; csvrow=comparison_idx[(rid,ti,field)]
                    for key,val in calculated.items():
                        loc=rid+"/"+str(ti)+"/"+field+"/"+key
                        if isinstance(val,Q):a.equal_number(existing[key],val,"comparison_summary",loc);a.equal_number(csvrow[key],val,"comparison_csv",loc)
                        else:
                            a.check("comparison_summary",existing[key]==val,loc)
                            parsed=json.loads(csvrow[key]) if isinstance(val,list) else csvrow[key]=="True" if isinstance(val,bool) else int(csvrow[key]) if isinstance(val,int) else csvrow[key]
                            a.check("comparison_csv",parsed==val,loc)
                    for ci,(actual,v) in enumerate(zip(existing["control_values_in_source_order"],vv)):a.equal_number(actual,v,"comparison_control_value",rid+"/"+str(ti)+"/"+field+"/"+str(ci))
                    a.check("comparison_support",existing["candidate_status"]==csvrow["candidate_status"]=="finite" and len(existing["control_values_in_source_order"])==existing["declared_controls"]==existing["observed_controls"]==int(csvrow["declared_controls"])==int(csvrow["observed_controls"])==16,rid+"/"+str(ti)+"/"+field)
                    for kind,count in [("finite",16),("missing",0),("null",0),("nonfinite",0),("nonnumeric",0)]:a.check("comparison_support_count",existing["support"][kind]==int(csvrow["n_"+kind])==count,rid+"/"+str(ti)+"/"+field+"/"+kind)
                    for sign,count in [("negative",sum(v<0 for v in vv)),("zero",sum(v==0 for v in vv)),("positive",sum(v>0 for v in vv))]:a.check("comparison_signed_counts",existing["control_sign_counts"][sign]==int(csvrow["control_"+sign])==count,rid+"/"+str(ti)+"/"+field+"/"+sign)
                    results["comparisons"].append({"run":rid,"time_index":ti,"step":step,"field":field,**calculated,"controls":vv})
                    if not within:results["exceptions"].append({"run":rid,"time_index":ti,"step":step,"field":field,"candidate":x,"control_min":ordered[0],"control_max":ordered[-1]})
            union=set().union(*map(set,control_sets))
            a.check("control_sets_change",reported["same_control_set_at_every_time"] is False and any(set(s)!=set(control_sets[0]) for s in control_sets[1:]),rid)
            results["controls"].append({"run":rid,"distinct_units":len(union),"units_by_time":control_sets,"adjacent_set_intersections":[len(set(x)&set(y)) for x,y in zip(control_sets,control_sets[1:])]})
            final_candidate=next(r for r in causal["rows"][-1]["ablations"] if r["is_candidate"])
            source_numeric={"final broad AUROC":lineage["targets"]["broad"]["test_auc"],
                "final hard-road AUROC":lineage["targets"]["broad"]["road_vehicle_auc"],
                "final ambulance AUROC":lineage["targets"]["ambulance"]["test_auc"],
                "final causal AUROC before":causal["rows"][-1]["specialist_auc_before"],
                "final causal AUROC after":final_candidate["specialist_auc_after"],
                "final response decrease SD":final_candidate["specialist_activation_change_sd_ambulance"]}
            for item in reported["numeric_report_reconciliation"]:
                a.equal_number(item["exported_or_recalculated_value"],source_numeric[item["field"]],"reported_value_source_binding",rid+"/"+item["field"])
                a.check("reported_precision",abs(number(item["exported_or_recalculated_value"])-number(item["reported_decimal"]))<=number(item["half_last_decimal_unit"]) and item["agrees_at_report_precision"],rid+"/"+item["field"])
            table1=next(r for r in reported["source_report_rows"] if len(r)==6)
            table2=next(r for r in reported["source_report_rows"] if len(r)==4)
            literal_numbers={"final broad AUROC":table1[3],"final hard-road AUROC":table1[4],"final ambulance AUROC":table1[5],
                "final causal AUROC before":table2[2].split("→")[0].strip(),"final causal AUROC after":table2[2].split("→")[1].strip(),"final response decrease SD":table2[3]}
            for label,text in literal_numbers.items():
                normalized=text.replace("−","-").replace("+","")
                digits=len(normalized.split(".")[1]);tol=Q(1,2*10**digits)
                a.check("report_table_numeric_rounding",abs(number(source_numeric[label])-Q(normalized))<=tol,rid+"/"+label)
            for column,target in [(1,"broad"),(2,"ambulance")]:
                own=next(r for r in results["onsets"] if r["run"]==rid and r["target"]==target)
                ii=own["first_pair_start_index"]
                rounded=[int(number(lineage["rows"][j]["epoch"])+Q(1,2)) for j in [ii-1,ii]] if ii is not None and ii>0 else None
                a.check("report_onset_bracket",[int(x) for x in re.findall(r"\d+",table1[column])]==rounded,rid+"/"+target)
            name=re.search(r"S(\d+)\.B(\d+)\.N(\d+), (negative|positive) mean",table2[1])
            expected_name=("mean-" if name and name[4]=="negative" else "mean+")+":stage"+(name[1] if name else "?")+".block"+(name[2] if name else "?")+"."+(name[3] if name else "?")
            a.check("report_candidate_coordinate",expected_name==candidate["coordinate"],rid)
            results["run_summaries"].append({"run":rid,"initial_readout_auc":metrics["rows"][0]["readout_auc"],"final_broad_auc":lineage["targets"]["broad"]["test_auc"],"final_hard_road_auc":lineage["targets"]["broad"]["road_vehicle_auc"],"source_selection":lineage["selection"],"source_control_rule":causal["controls"]})
        a.check("all_nine_onsets",len(results["onsets"])==9)
        a.check("all120_comparisons",len(results["comparisons"])==120)
        for field in FIELDS:
            subset=[r for r in results["comparisons"] if r["field"]==field]
            a.check("all15_field_times",len(subset)==15,field)
        results["range_counts"]={field:sum(r["candidate_in_closed_control_range"] for r in results["comparisons"] if r["field"]==field) for field in FIELDS}
        a.check("response_range_claim",results["range_counts"]["specialist_activation_change_sd_ambulance"]==14)
        a.check("auroc_range_claim",results["range_counts"]["specialist_auc_after"]==results["range_counts"]["specialist_auc_decrease"]==14)
        a.check("control_union_claim",[r["distinct_units"] for r in results["controls"]]==[46,47,49])
        readouts=[number(r["initial_readout_auc"]) for r in results["run_summaries"]]
        a.check("report_initial_readout_range",abs(min(readouts)-Q(855,1000))<=Q(5,10000) and abs(max(readouts)-Q(874,1000))<=Q(5,10000))
        if not a.numeric:
            # Read-only reconciliation of provenance receipts; no live retrieval.
            provenance=a.load("input_provenance.json");allow=a.load("retrieval_allowlist.json")
            a.check("retrieval_allowlist",len(allow["requests"])==allow["additional_retrieval_cap"]==6)
            observed=[r for r in provenance["records"] if r["mode"]=="additional_public_GET"]
            a.check("six_recorded_retrievals",len(observed)==6 and {r["url"] for r in observed}=={r["url"] for r in allow["requests"]})
            for record in provenance["records"]:
                actual=a.base/record["path"]
                a.check("provenance_file_identity",sha(actual)==record["sha256"] and actual.stat().st_size==record["bytes"],record["path"])
                if record["mode"]=="additional_public_GET":a.check("recorded_http_success",record["status"]==200 and record["url"]==record["final_url"],record["path"])
                else:
                    original=a.base.parent.parent/record["original_path"]
                    a.check("unchanged_prior_copy",original.exists() and sha(original)==record["sha256"],record["original_path"])
            corpus=a.load("source_corpus_comparison.json")
            old_hashes=set()
            for record in corpus["original_files"]:
                original=a.base.parent.parent/record["path"];old_hashes.add(record["sha256"])
                a.check("older_corpus_identity",original.exists() and sha(original)==record["sha256"] and original.stat().st_size==record["bytes"],record["path"])
            for record in corpus["exported_measurement_files"]:a.check("measurement_corpus_nonidentity",sha(a.base/record["path"])==record["sha256"] and record["sha256"] not in old_hashes and record["matches_original_source_file"] is False,record["path"])
            # The report is read locally; it is not reproduced or required in numerical-only mode.
            report=(a.base/"inputs/RESULTS.md").read_text()
            source_table=[[s.strip() for s in line.strip("|").split("|")] for line in report.splitlines() if line.startswith(("| Tiny,","| Wider,"))]
            preserved=[row for run in saved["runs"] for row in run["source_report_rows"]]
            a.check("source_report_table_transcription",sorted(source_table)==sorted(preserved))
            a.check("source_protocol_boundary","Frozen before inspecting the new measurements, 2026-09-20." in (a.base/"inputs/PROTOCOL.md").read_text())
        for relative,identity in a.identities.items():a.check("inputs_unchanged_after_replay",sha(a.base/relative)==identity["sha256"],relative)
    except Exception as exc:error=repr(exc);a.failures.append({"category":"execution_exception","error":error})
    receipt={"passed":not a.failures,"checks":sum(a.counts.values()),"checks_by_category":dict(sorted(a.counts.items())),"failures":a.failures,
             "utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-start,"reviewer_code_sha256":sha(Path(__file__)),
             "reviewed_manifest_sha256":MANIFEST_SHA,"mode":"numerical_only" if a.numeric else "full_frozen_audit",
             "omitted_original_bound_files":a.omitted,"input_hashes":a.identities,
             "scope":"Independent rational arithmetic from written exported values; source HTTP/selection statements not authenticated experimentally.",
             "new_network_requests":0,"model_forwards":0,"fits":0,"bootstrap_draws":0,"raw_response_auroc_computations":0}
    for name,value in [("receipt.json",receipt),("recomputed.json",results)]:
        with (a.output/name).open("x") as f:json.dump(clean(value),f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:receipt[k] for k in ["passed","checks","failures","mode","elapsed_seconds"]}))
    if a.failures:raise SystemExit(1)


if __name__=="__main__":main()
