#!/usr/bin/env python3
"""Exact, standalone SAT semantic controls. Standard library only.

No neural code, weights, inference, pilot data, or outcome files are loaded.
Clause IDs here are local canonical tuples, not checkpoint vocabulary IDs.
"""
from collections import defaultdict
from itertools import combinations, product
from math import comb
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def grammar(n):
    # A literal (j,b) is true iff variable j has value b.
    return [(i, a, j, b) for i in range(n) for j in range(i + 1, n)
            for a, b in product((0, 1), repeat=2)]


def holds(clause, valuation):
    i, a, j, b = clause
    return ((valuation >> i) & 1) == a or ((valuation >> j) & 1) == b


class Oracle:
    def __init__(self, n):
        self.n = n
        self.clauses = grammar(n)
        self.universe = (1 << (1 << n)) - 1
        self.masks = [sum(1 << v for v in range(1 << n) if holds(c, v))
                      for c in self.clauses]
        self.ones = [sum(1 << v for v in range(1 << n) if v & (1 << j))
                     for j in range(n)]

    def model_mask(self, formula):
        result = self.universe
        for c in formula:
            result &= self.masks[c]
        return result

    def boolean_mask(self, formula):
        return sum(1 << v for v in range(1 << self.n)
                   if all(holds(self.clauses[c], v) for c in formula))

    def counts(self, formula):
        return [sum(not holds(self.clauses[c], v) for c in formula)
                for v in range(1 << self.n)]

    def backbone(self, models):
        assert models
        return tuple(1 if models & (~one & self.universe) == 0 else
                     0 if models & one == 0 else -1 for one in self.ones)

    def predicts_unsat(self, backbone, new_clause):
        i, a, j, b = self.clauses[new_clause]
        return backbone[i] == 1 - a and backbone[j] == 1 - b

    def profile(self, formula):
        assert self.model_mask(formula)
        return tuple(self.backbone(self.model_mask(formula[:i] + formula[i + 1:]))
                     for i in range(len(formula)))

    def menu(self, formula, slot):
        rest = self.model_mask(formula[:slot] + formula[slot + 1:])
        return tuple(bool(rest & self.masks[new]) for new in range(len(self.clauses))
                     if new != formula[slot])


def backbone_key(backbone):
    return backbone if sum(v >= 0 for v in backbone) >= 2 else None


def profile_encode(profile):
    """Each variable: unknown throughout, or value and nonempty slot mask."""
    m, n = len(profile), len(profile[0])
    cap = (1 << m) - 1
    result = []
    for j in range(n):
        fixed = {b[j] for b in profile if b[j] >= 0}
        assert len(fixed) <= 1
        mask = sum(1 << i for i, b in enumerate(profile) if b[j] >= 0)
        result.append(0 if not mask else mask + next(iter(fixed)) * cap)
    return result


def profile_decode(codes, slot, m):
    cap = (1 << m) - 1
    result = []
    for code in codes:
        if code == 0:
            result.append(-1)
        else:
            value, mask = (0, code) if code <= cap else (1, code - cap)
            result.append(value if mask & (1 << slot) else -1)
    return tuple(result)


def exhaustive_small():
    oracle = Oracle(3)
    observed = defaultdict(dict)
    formulas = sat_formulas = menus = actions = 0
    for formula in product(range(12), repeat=4):
        formulas += 1
        model_mask = oracle.model_mask(formula)
        assert model_mask == oracle.boolean_mask(formula)
        if not model_mask:
            continue
        sat_formulas += 1
        profile = oracle.profile(formula)
        encoded = profile_encode(profile)
        for slot, old in enumerate(formula):
            backbone = profile[slot]
            assert profile_decode(encoded, slot, 4) == backbone
            actual = oracle.menu(formula, slot)
            predicted = tuple(not oracle.predicts_unsat(backbone, new)
                              for new in range(12) if new != old)
            assert actual == predicted
            key = backbone_key(backbone)
            previous = observed[(slot, old)].setdefault(key, actual)
            assert previous == actual
            menus += 1
            actions += 11
    for contexts in observed.values():
        assert len(set(contexts.values())) == len(contexts)
    return {"variables": 3, "clauses": 4, "ordered_formulas_checked": formulas,
            "initially_sat_formulas": sat_formulas, "menus_checked": menus,
            "replacement_labels_checked": actions,
            "check": "Boolean/bitset oracle agreement; backbone menu theorem; exact menu equivalence; profile coding"}


def clause_id(oracle, first, second):
    (i, a), (j, b) = sorted((first, second))
    assert i != j
    return oracle.clauses.index((i, a, j, b))


def realise_backbone(oracle, target, old, slots=10):
    fixed = [(j, value) for j, value in enumerate(target) if value >= 0]
    if not fixed:
        rest = [old]
    else:
        anchor = fixed[0]
        auxiliary = next(j for j in range(oracle.n) if j != anchor[0])
        rest = [clause_id(oracle, anchor, (auxiliary, 0)),
                clause_id(oracle, anchor, (auxiliary, 1))]
        neg_anchor = (anchor[0], 1 - anchor[1])
        rest.extend(clause_id(oracle, neg_anchor, item) for item in fixed[1:])
    assert len(rest) <= slots - 1
    rest.extend([rest[0]] * (slots - 1 - len(rest)))
    formula = tuple([old] + rest)
    assert oracle.model_mask(formula)
    assert oracle.backbone(oracle.model_mask(tuple(rest))) == target
    return formula


def conditional_class_count():
    oracle = Oracle(5)
    contexts = []
    for old in range(40):
        menus = set()
        targets = 0
        for target in product((-1, 0, 1), repeat=5):
            if oracle.predicts_unsat(target, old):
                continue
            formula = realise_backbone(oracle, target, old)
            menus.add(oracle.menu(formula, 0))
            targets += 1
        assert targets == 216 and len(menus) == 206
        contexts.append({"old_clause": oracle.clauses[old], "compatible_backbones": targets,
                         "distinct_menus": len(menus)})
    return {"variables": 5, "formula_clauses": 10, "old_clause_contexts_checked": 40,
            "compatible_backbone_realisations": 40 * 216,
            "distinct_menu_classes_per_context": 206,
            "minimum_fixed_length_bits_for_one_fixed_context": (206 - 1).bit_length(),
            "contexts": contexts}


def multiset_rank(values):
    return sum(comb(v + i, i + 1) for i, v in enumerate(sorted(values)))


def multiset_unrank(rank, types, slots):
    assert 0 <= rank < comb(types + slots - 1, slots)
    b = [0] * slots
    upper = types + slots - 2
    remainder = rank
    for i in range(slots, 0, -1):
        low, high = i - 1, upper
        while low < high:
            mid = (low + high + 1) // 2
            if comb(mid, i) <= remainder:
                low = mid
            else:
                high = mid - 1
        b[i - 1] = low
        remainder -= comb(low, i)
        upper = low - 1
    assert remainder == 0
    return tuple(b[i] - i for i in range(slots))


def multiset_checks():
    small_total = comb(12 + 4 - 1, 4)
    for rank in range(small_total):
        values = multiset_unrank(rank, 12, 4)
        assert len(values) == 4 and all(0 <= x < 12 for x in values)
        assert multiset_rank(values) == rank
    total = comb(49, 10)
    for i in range(1001):
        rank = i * (total - 1) // 1000
        values = multiset_unrank(rank, 40, 10)
        assert multiset_rank(values) == rank
    return {"types": 40, "slots": 10, "multiset_count": total,
            "fixed_length_bits": (total - 1).bit_length(),
            "small_domain_all_ranks_checked": small_total,
            "full_domain_deterministic_rank_checks": 1001,
            "scope": "exact SAT-semantic multiset; cannot recover token order or checkpoint computation"}


def horizon_witnesses():
    oracle = Oracle(5)
    c = clause_id(oracle, (0, 1), (2, 1))  # x0 OR x2
    d = clause_id(oracle, (0, 1), (2, 0))  # x0 OR NOT x2
    e = clause_id(oracle, (0, 0), (1, 1))  # NOT x0 OR x1
    new = clause_id(oracle, (0, 0), (1, 0))  # NOT x0 OR NOT x1
    records = []
    for horizon in range(1, 5):
        f = tuple([c] * horizon + [d] + [e] * (9 - horizon))
        g = tuple(list(f[:-1]) + [c])
        assert len(f) == len(g) == 10
        qf, qg = oracle.counts(f), oracle.counts(g)
        capf = [min(q, horizon) for q in qf]
        capg = [min(q, horizon) for q in qg]
        assert capf == capg
        assert oracle.model_mask(f) == oracle.model_mask(g) != 0
        assert oracle.model_mask(f) == oracle.boolean_mask(f)
        assert oracle.model_mask(g) == oracle.boolean_mask(g)
        original_f, original_g = f, g
        actions = []
        path = []
        for slot in range(horizon):
            replacement = d if slot < horizon - 1 else new
            assert f[slot] == g[slot] == c and replacement != c
            actions.append({"slot": slot, "old": oracle.clauses[c],
                            "new": oracle.clauses[replacement]})
            f = f[:slot] + (replacement,) + f[slot + 1:]
            g = g[:slot] + (replacement,) + g[slot + 1:]
            satf, satg = bool(oracle.boolean_mask(f)), bool(oracle.boolean_mask(g))
            path.append([satf, satg])
            if slot < horizon - 1:
                assert satf and satg
            # Combinatorial-code update is exact for this valid descriptor.
            for before, after in ((original_f, f), (original_g, g)):
                decoded = list(multiset_unrank(multiset_rank(before), 40, 10))
                for action in actions:
                    old_id = oracle.clauses.index(tuple(action["old"]))
                    new_id = oracle.clauses.index(tuple(action["new"]))
                    decoded.remove(old_id)
                    decoded.append(new_id)
                assert sorted(decoded) == sorted(after)
                assert oracle.model_mask(tuple(decoded)) == oracle.model_mask(after)
        assert path[-1] == [True, False]
        record = {"horizon": horizon, "formula_f": [oracle.clauses[v] for v in original_f],
                  "formula_g": [oracle.clauses[v] for v in original_g],
                  "q_f": qf, "q_g": qg, "identical_cap": horizon,
                  "common_current_models": [v for v in range(32) if oracle.model_mask(original_f) & (1 << v)],
                  "actions": actions, "sat_path_f_g": path,
                  "all_deletion_backbone_profiles_equal": oracle.profile(original_f) == oracle.profile(original_g)}
        if horizon == 1:
            record["deletion_backbone_f"] = oracle.profile(original_f)[0]
            record["deletion_backbone_g"] = oracle.profile(original_g)[0]
        if horizon >= 2:
            assert record["all_deletion_backbone_profiles_equal"]
        records.append(record)
    return records


def decision_witness():
    oracle = Oracle(5)
    c = clause_id(oracle, (0, 1), (1, 1))
    force_not_x1 = [clause_id(oracle, (1, 0), (4, value)) for value in (0, 1)]
    force_x2 = [clause_id(oracle, (2, 1), (4, value)) for value in (0, 1)]
    force_x3 = [clause_id(oracle, (3, 1), (4, value)) for value in (0, 1)]
    x0_implies_x3 = clause_id(oracle, (0, 0), (3, 1))
    x0_implies_x2 = clause_id(oracle, (0, 0), (2, 1))
    f = tuple([c] + force_not_x1 + force_x2 + [x0_implies_x3] + [force_not_x1[0]] * 4)
    g = tuple([c] + force_not_x1 + force_x3 + [x0_implies_x2] + [force_not_x1[0]] * 4)
    assert len(f) == len(g) == 10
    common = oracle.model_mask(f)
    assert common == oracle.model_mask(g) == oracle.boolean_mask(f) == oracle.boolean_mask(g)
    assert common.bit_count() == 2
    b_f, b_g = oracle.profile(f)[0], oracle.profile(g)[0]
    assert b_f == (-1, 0, 1, -1, -1) and b_g == (-1, 0, -1, 1, -1)
    success_f = [new for new in range(40) if new != c and
                 not oracle.boolean_mask((new,) + f[1:])]
    success_g = [new for new in range(40) if new != c and
                 not oracle.boolean_mask((new,) + g[1:])]
    assert len(success_f) == len(success_g) == 1 and success_f != success_g
    return {"formula_f": [oracle.clauses[v] for v in f],
            "formula_g": [oracle.clauses[v] for v in g],
            "context": {"slot": 0, "old": oracle.clauses[c]},
            "common_current_models": [v for v in range(32) if common & (1 << v)],
            "deletion_backbone_f": b_f, "deletion_backbone_g": b_g,
            "only_unsat_replacement_f": oracle.clauses[success_f[0]],
            "only_unsat_replacement_g": oracle.clauses[success_g[0]],
            "uniform_pair_optimal_success_with_truth_table_report": "1/2",
            "each_state_oracle_success": 1,
            "scope": "symbolic semantic witness, no neural report or checkpoint claim"}


def exhaustive_horizon_control():
    oracle = Oracle(2)
    counts = {1: 0, 2: 0, 3: 0}
    for original in product(range(4), repeat=3):
        q = oracle.counts(original)
        for horizon in (1, 2, 3):
            initial = tuple(min(v, horizon + 1) for v in q)
            def visit(formula, state, remaining):
                if not remaining:
                    assert any(v == 0 for v in state) == bool(oracle.boolean_mask(formula))
                    counts[horizon] += 1
                    return
                for slot, old in enumerate(formula):
                    for new in range(4):
                        if new == old:
                            continue
                        # Remaining horizon drops, so the sufficient cap drops too.
                        updated = tuple(min(state[v] - int(not holds(oracle.clauses[old], v))
                                            + int(not holds(oracle.clauses[new], v)), remaining)
                                        for v in range(4))
                        assert min(updated) >= 0
                        new_formula = formula[:slot] + (new,) + formula[slot + 1:]
                        visit(new_formula, updated, remaining - 1)
            visit(original, initial, horizon)
    return {"variables": 2, "slots": 3, "initial_formulas": 64,
            "all_valid_action_paths_checked_by_horizon": counts,
            "check": "cap at remaining-horizon plus one; independently evaluated final Boolean SAT"}


def main():
    result = {"status": "PASS", "neural_data_or_outcomes_read": False,
              "protected_data_read": False, "random_search": False,
              "local_clause_notation": "(variable index, satisfying value, variable index, satisfying value)",
              "exhaustive_one_step": exhaustive_small(),
              "exact_conditional_quotient": conditional_class_count(),
              "multiset_control": multiset_checks(),
              "same_truth_table_conflicting_optimal_actions": decision_witness(),
              "horizon_separation_witnesses": horizon_witnesses(),
              "exhaustive_bounded_horizon": exhaustive_horizon_control()}
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, path in (("prior_art_boundary", ROOT.parent / "next_phase/sat_prior_art_boundary.md"),
                       ("frozen_contract", ROOT.parent / "sat_case/contract.md")):
        result[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "exhaustive_one_step",
                                            "multiset_control", "exhaustive_bounded_horizon")}))


if __name__ == "__main__":
    main()
