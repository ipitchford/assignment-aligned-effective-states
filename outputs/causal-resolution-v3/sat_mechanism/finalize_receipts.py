"""Post-extraction CLERICAL correction only; no extraction, fitting or metrics.

The frozen finalize function calls the integer st_size attribute as a function.
Replace exactly that accessor in an in-memory copy of that function. Preserve
all frozen files and the ERROR_STOP receipt. Then format already-saved metrics.
"""
import ast
import csv
import hashlib
import json
import analyze
from protocol import *


def main():
    frozen()
    if (ROOT/"finalization_receipt.json").exists():raise RuntimeError("Clerical finalization already completed")
    result_names=["stage_a.json","stage_b.json","original_native.npz","edited_native.npz","stage_b_raw.npz","bootstrap_raw.npz","replay.json","execution.json"]
    before={name:sha(ROOT/name) for name in result_names}
    tree=ast.parse((ROOT/"analyze.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="finalize")
    changes=[]
    class FixAccessor(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=="st_size":
                assert not node.args and not node.keywords
                changes.append(dict(line=node.lineno,before=".stat().st_size()",after=".stat().st_size"))
                return ast.copy_location(node.func,node)
            return node
    fn=FixAccessor().visit(fn);assert len(changes)==1
    corrected=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace=analyze.__dict__.copy()
    exec(compile(corrected,"<clerical-finalizer-only>","exec"),namespace)
    namespace["finalize"]()
    assert before=={name:sha(ROOT/name) for name in result_names}
    a=read_json("stage_a.json");b=read_json("stage_b.json")
    columns=["unit","assignment","stage_a_assessment_q_one","stage_a_assessment_q_many","stage_a_pre_auroc","stage_a_pre_threshold_balanced_accuracy","readiness_threshold","threshold_float32_le_hex","stage_b_clipped_assessment_q_one","stage_b_clipped_assessment_q_many","native_structured_balanced_accuracy","native_post_score_balanced_accuracy","native_mismatched_unit_balanced_accuracy","native_mismatched_assignment_balanced_accuracy","native_context_balanced_accuracy","native_symbolic_oracle_balanced_accuracy","repair_minus_neutral_q_one","repair_minus_neutral_q_many","interaction","negative_D_cases"]
    with (ROOT/"unit_summary.csv").open("w",newline="") as out:
        writer=csv.writer(out);writer.writerow(columns)
        for ua,ub in zip(a["units"],b["units"]):
            assert ua["unit"]==ub["unit"]
            x=ua["assessment"];z=ub["primary_assessment"];rule=ua["pre_rule"]
            writer.writerow([ua["unit"],ua["assignment"],x["q_one"],x["q_many"],x["pre_auroc"],x["pre_threshold_metrics"]["balanced_accuracy"],rule["threshold"] if rule["kind"]=="threshold" else "constant_"+str(rule["value"]),rule["threshold_float32_le_hex"],z["effect"]["q_one_n"],z["effect"]["q_many_n"],*[z["models"][m]["balanced_accuracy"] for m in MODELS],z["effect"]["q_one_mean_D"],z["effect"]["q_many_mean_D"],z["effect"]["interaction"],z["effect"]["negative_D_cases"]])
    intervals=b["bootstrap"]["intervals"]
    with (ROOT/"bootstrap_summary.csv").open("w",newline="") as out:
        writer=csv.writer(out);writer.writerow(["metric","valid_draws","na_draws","lower_2_5","upper_97_5"])
        summaries={"interaction":intervals["interaction"]}
        summaries.update({"BA_"+k:v for k,v in intervals["native_balanced_accuracy"].items()})
        summaries.update({"structured_minus_"+k:v for k,v in intervals["structured_minus_control"].items()})
        for name,entry in summaries.items():writer.writerow([name,entry["valid_draws"],entry["na_draws"],*(entry["interval_2_5_97_5"] or [None,None])])
    with (ROOT/"REPORT.md").open("a") as out:
        out.write("\nThe frozen driver completed all scientific analyses and the separate-process replay, then stopped on a clerical file-size accessor typo in final report assembly (`st_size()` instead of `st_size`). The frozen source, exception log and `execution.json` ERROR_STOP state are preserved. `finalize_receipts.py` corrects only that accessor in an in-memory copy of the reporting function, without inference, fitting, new statistics, or changes to any scientific receipt. It also formats existing metrics into CSV. `finalization_receipt.json` records this post-extraction clerical correction.\n")
    receipt=dict(status="CLERICAL_FINALIZATION_COMPLETE",scientific_execution_status="A,B,and16forwardreplaycompletedoncebeforeformattingerror",preserved_original_driver_status=read_json("execution.json")["status"],original_error="TypeError: int object is not callable at .stat().st_size()",frozen_analyze_sha256=sha(ROOT/"analyze.py"),correction_script_sha256=sha(__file__),correction=changes,corrected_finalizer_ast_sha256=hashlib.sha256(ast.dump(corrected,include_attributes=False).encode()).hexdigest(),scientific_receipts_unchanged=before,new_native_forwards=0,new_fits=0,new_statistical_analyses=0,outputs={name:sha(ROOT/name) for name in ["cost_ledger.json","summary.json","REPORT.md","unit_summary.csv","bootstrap_summary.csv"]})
    write_json("finalization_receipt.json",receipt)
    print(json.dumps(dict(status=receipt["status"],preserved_driver_status=receipt["preserved_original_driver_status"],new_native_forwards=0,new_fits=0)))


if __name__=="__main__":main()
