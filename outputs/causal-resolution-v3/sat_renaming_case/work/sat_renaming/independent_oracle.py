"""Ordinary-loop Boolean oracle for the fixed renaming assay.

This module imports neither NumPy nor the sampler, prior oracle, model or runtime.
Token IDs and clause numbering are decoded from the native grammar directly.
It has no random generator, model loader, file writer or import-time computation
on formulas. The verifier is responsible for authorizing which saved inputs it
evaluates.
"""
import hashlib
import operator

VARIABLE_PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
                  (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
UNIT_ORDER = (261, 291, 411, 492)
ASSIGNMENT_TEXT = ("TTFTT", "TFTTT", "TTFFT", "TFTFT")
ACTION_FIELDS = ("slot", "old_clause_id", "kept_literal", "new_variable", "new_clause_id")


def integer(value):
    """Reject coercion from float/string instead of silently truncating it."""
    if isinstance(value, bool):
        raise ValueError("Boolean supplied where an integer token/index is required")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError("Non-integer token/index") from exc


def assignment_bits(index):
    index = integer(index)
    if not 0 <= index < 32:
        raise ValueError("Assignment index outside 0..31")
    return tuple(bool(index & (1 << (4 - variable))) for variable in range(5))


def assignment_index(text):
    if len(text) != 5 or any(letter not in "TF" for letter in text):
        raise ValueError("Expected five T/F characters in x0..x4 order")
    result = 0
    for letter in text:
        result = 2 * result + (letter == "T")
    return result


def literal_value(token, assignment):
    token = integer(token)
    if not 0 <= token < 10 or len(assignment) != 5:
        raise ValueError("Invalid literal or assignment")
    value = bool(assignment[token % 5])
    return value if token < 5 else not value


def clause_value(clause, assignment):
    return literal_value(clause[0], assignment) or literal_value(clause[1], assignment)


def validate_clause(clause, restricted=False):
    values = tuple(integer(token) for token in clause)
    if len(values) != 2 or any(token < 0 or token >= 10 for token in values):
        raise ValueError("A clause must contain two literal tokens")
    variables = tuple(token % 5 for token in values)
    if not variables[0] < variables[1]:
        raise ValueError("Clause variables must be distinct and in increasing order")
    if restricted and variables == (1, 2):
        raise ValueError("Forbidden variable pair {x1,x2}")
    return values


def clause_from_id(clause_id, restricted=False):
    clause_id = integer(clause_id)
    if not 0 <= clause_id < 40:
        raise ValueError("Clause ID outside original 0..39 numbering")
    pair_index, signs = divmod(clause_id, 4)
    left, right = VARIABLE_PAIRS[pair_index]
    left += 5 * (signs // 2)
    right += 5 * (signs % 2)
    return validate_clause((left, right), restricted)


def clause_to_id(clause, restricted=False):
    left, right = validate_clause(clause, restricted)
    pair_index = VARIABLE_PAIRS.index((left % 5, right % 5))
    return 4 * pair_index + 2 * (left >= 5) + (right >= 5)


def decode_tokens(tokens, restricted=False):
    tokens = tuple(integer(token) for token in tokens)
    if len(tokens) != 41 or tokens[-1] != 12:
        raise ValueError("Expected 40 clause tokens followed by colon token 12")
    clauses = []
    for slot in range(10):
        offset = 4 * slot
        if tokens[offset] != 10 or tokens[offset + 3] != 11:
            raise ValueError("Invalid clause delimiter tokens")
        clauses.append(validate_clause(tokens[offset + 1:offset + 3], restricted))
    return tuple(clauses)


def tokens_from_clauses(clauses, restricted=False):
    clauses = tuple(clauses)
    if len(clauses) != 10:
        raise ValueError("Expected ten ordered clause occurrences")
    result = []
    for clause in clauses:
        left, right = validate_clause(clause, restricted)
        result.extend((10, left, right, 11))
    result.append(12)
    return tuple(result)


def tokens_from_clause_ids(clause_ids, restricted=False):
    return tokens_from_clauses((clause_from_id(cid, restricted) for cid in clause_ids), restricted)


def clause_ids_from_tokens(tokens, restricted=False):
    return tuple(clause_to_id(clause, restricted) for clause in decode_tokens(tokens, restricted))


def evaluate_tokens(tokens, restricted=False):
    """Evaluate each literal, clause occurrence and valuation using Boolean loops."""
    clauses = decode_tokens(tokens, restricted)
    counts, truths = [], []
    for valuation in range(32):
        assignment = assignment_bits(valuation)
        violations = 0
        satisfied = True
        for clause in clauses:
            value = clause_value(clause, assignment)
            if not value:
                violations += 1
            satisfied = satisfied and value
        counts.append(violations)
        truths.append(satisfied)
    solutions = tuple(index for index, value in enumerate(truths) if value)
    return {"counts": tuple(counts), "truth_table": tuple(truths),
            "solutions": solutions, "label": 13 if solutions else 14}


def truth_table(tokens):
    return evaluate_tokens(tokens)["truth_table"]


def violation_counts(tokens):
    return evaluate_tokens(tokens)["counts"]


def rename_variable(variable):
    variable = integer(variable)
    if not 0 <= variable < 5:
        raise ValueError("Variable outside x0..x4")
    return 2 if variable == 1 else 1 if variable == 2 else variable


def rename_literal(token):
    token = integer(token)
    if not 0 <= token < 10:
        raise ValueError("Literal outside 0..9")
    return rename_variable(token % 5) + (5 if token >= 5 else 0)


def rename_assignment(index):
    original = assignment_bits(index)
    renamed = list(original)
    renamed[1], renamed[2] = original[2], original[1]
    result = 0
    for value in renamed:
        result = 2 * result + value
    return result


def rename_clause(clause, require_position_preservation=False):
    mapped = tuple(rename_literal(token) for token in validate_clause(clause))
    canonical = tuple(sorted(mapped, key=lambda token: token % 5))
    if require_position_preservation and mapped != canonical:
        raise ValueError("Renaming changed a literal position")
    return canonical


def rename_tokens(tokens, require_position_preservation=False):
    clauses = decode_tokens(tokens, restricted=require_position_preservation)
    return tokens_from_clauses(
        (rename_clause(clause, require_position_preservation) for clause in clauses),
        restricted=require_position_preservation)


def normalized_action(action):
    if set(action) != set(ACTION_FIELDS):
        raise ValueError("Action must contain all and only the five frozen descriptor fields")
    values = {name: integer(action[name]) for name in ACTION_FIELDS}
    if not 0 <= values["slot"] < 10:
        raise ValueError("Action slot outside 0..9")
    old = clause_from_id(values["old_clause_id"], restricted=True)
    new = clause_from_id(values["new_clause_id"], restricted=True)
    kept = values["kept_literal"]
    variable = values["new_variable"]
    if not 0 <= variable < 5 or kept not in old or kept not in new:
        raise ValueError("Kept literal absent or invalid new variable")
    if variable in (old[0] % 5, old[1] % 5):
        raise ValueError("New variable is not outside the old clause")
    added = tuple(token for token in new if token != kept)
    if len(added) != 1 or added[0] % 5 != variable:
        raise ValueError("New clause does not retain the literal and replace the other variable")
    return values


def apply_action(tokens, action):
    action = normalized_action(action)
    clauses = list(decode_tokens(tokens, restricted=True))
    if clause_to_id(clauses[action["slot"]]) != action["old_clause_id"]:
        raise ValueError("Action old clause does not match the specified slot")
    clauses[action["slot"]] = clause_from_id(action["new_clause_id"], restricted=True)
    return tokens_from_clauses(clauses, restricted=True)


def rename_action(action):
    action = normalized_action(action)
    return normalized_action({
        "slot": action["slot"],
        "old_clause_id": clause_to_id(rename_clause(clause_from_id(action["old_clause_id"]), True)),
        "kept_literal": rename_literal(action["kept_literal"]),
        "new_variable": rename_variable(action["new_variable"]),
        "new_clause_id": clause_to_id(rename_clause(clause_from_id(action["new_clause_id"]), True)),
    })


def token_bytes(tokens):
    decode_tokens(tokens)
    return bytes(integer(token) for token in tokens)


def token_sha256(tokens):
    return hashlib.sha256(token_bytes(tokens)).hexdigest()


def orbit_bytes(tokens):
    return min(token_bytes(tokens), token_bytes(rename_tokens(tokens)))


def orbit_sha256(tokens):
    return hashlib.sha256(orbit_bytes(tokens)).hexdigest()
