"""Audit recovered published rules without loading model weights or pilot outcomes."""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import re

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "sat_case/upstream"


def evaluate(rule, table):
    op = rule["op"]
    if op == "phi":
        return table[rule["truth_table_index"]]
    if op == "not":
        return not evaluate(rule["arg"], table)
    vals = [evaluate(a, table) for a in rule["args"]]
    return all(vals) if op == "and" else any(vals)


def formula_table(clauses):
    # Literal [i, s] is true when variable i equals Boolean s.
    return [
        all(any(values[i] == sign for i, sign in clause) for clause in clauses)
        for values in itertools.product([False, True], repeat=5)
    ]


def unique_assignment_formula(assignment):
    # (l_i or l_j) AND (l_i or not l_j) forces l_i, with j=(i+1) mod5.
    values = [c == "T" for c in assignment]
    clauses = []
    for i in range(5):
        j = (i + 1) % 5
        for other_sign in [values[j], not values[j]]:
            clauses.append(sorted([[i, values[i]], [j, other_sign]]))
    return clauses


def main():
    mapping_path = OUT / "sat_published_neuron_rules.json"
    mapping = json.loads(mapping_path.read_text())
    notebook_path = SOURCE / "interpretation.ipynb"
    notebook = json.loads(notebook_path.read_text())
    output = notebook["cells"][149]["outputs"][0]["data"]["text/plain"]
    source_disjunctions = ast.literal_eval("".join(output))
    source_disjunctions = {
        k: [re.search(r"\[([TF]{5})\]", value).group(1) for value in values]
        for k, values in source_disjunctions.items()
    }
    assert len(mapping["rules"]) == 34
    for row in mapping["rules"]:
        assert row["disjunction_only_assignments"] == source_disjunctions[row["unit"]]

    missing_singletons = []
    for index in range(32):
        table = [i == index for i in range(32)]
        if not any(evaluate(row["general_rule_ast"], table) for row in mapping["rules"]):
            assignment = "".join("T" if b == "1" else "F" for b in f"{index:05b}")
            clauses = unique_assignment_formula(assignment)
            assert len(clauses) == 10
            assert all(len(c) == 2 and c[0][0] < c[1][0] for c in clauses)
            assert formula_table(clauses) == table
            missing_singletons.append({
                "assignment": assignment,
                "truth_table_index": index,
                "clauses": clauses,
                "table2_or": False,
                "oracle_sat": True,
                "neural_model_evaluated": False,
            })

    disjunction_union = set().union(*(set(v) for v in source_disjunctions.values()))
    assert len(disjunction_union) == 32
    # The general table's omission is a symbolic fact, not a neural inference result.
    assert [x["assignment"] for x in missing_singletons] == ["TFTFF", "TTTFT"]
    unit_bindings = []
    for i, cell in enumerate(notebook["cells"]):
        code = "".join(cell.get("source", []))
        if re.search(r"(^|\n)(for unit\b|unit\s*=)", code):
            unit_bindings.append({
                "cell_index_zero_based": i,
                "execution_count": cell.get("execution_count"),
                "binding": code.splitlines()[0] if i == 88 else "for unit in units_for_sat.numpy():",
            })

    receipt = {
        "kind": "source-and-symbolic-only audit; no model weights or pilot outcomes loaded",
        "date": "2026-10-15",
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "rules": 34,
        "all_table3_rows_match_notebook_cell149": True,
        "table3_assignment_coverage": len(disjunction_union),
        "table2_single_atom_units": mapping["single_assignment_general_units"],
        "table2_singleton_truth_table_counterexamples": missing_singletons,
        "notebook_global_unit_bindings": unit_bindings,
        "notebook_clause_coefficients_cell": {
            "index_zero_based": 185,
            "execution_count": notebook["cells"][185]["execution_count"],
            "uses_variable": "unit",
            "inference_from_visible_execution_order": "unit is499 aftercell148; filenames in194/195 sayunit10",
            "limit": "Saved execution counters do not rule out unrecorded notebook state; stored coefficient outputs must not be attributed to unit10 without replay.",
        },
    }
    (OUT / "sat_source_rule_audit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "all_table3_rows_match_notebook": True,
        "rules": 34,
        "single_atom_units": len(mapping["single_assignment_general_units"]),
        "table2_missed_singletons": [x["assignment"] for x in missing_singletons],
        "pilot_outcomes_loaded": False,
        "model_weights_loaded": False,
    }))


if __name__ == "__main__":
    main()
