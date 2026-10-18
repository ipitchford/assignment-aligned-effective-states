#!/usr/bin/env python3
"""Post-GO integrity audit; does not alter the protected study or its gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    study = args.study_dir.resolve()
    sys.path.insert(0, str(study))
    os.environ["SAT_ALL17_FREEZE_SHA256"] = args.freeze_sha256

    import protocol
    from pilot_common import encode
    from verify_bank import decode, encode_formula, transform_formula

    frozen = protocol.verify_frozen_packet()
    construction = json.loads((study / "construction.json").read_text())
    verification = json.loads((study / "bank_verification.json").read_text())
    native_receipt = json.loads((study / "native_receipt.json").read_text())
    replay_receipt = json.loads((study / "replay_receipt.json").read_text())
    analysis = json.loads((study / "analysis.json").read_text())
    with np.load(study / "candidate_bank.npz", allow_pickle=False) as z:
        candidates = {key: z[key].copy() for key in z.files}
    with np.load(study / "selected_bank.npz", allow_pickle=False) as z:
        bank = {key: z[key].copy() for key in z.files}
    with np.load(study / "native_outputs.npz", allow_pickle=False) as z:
        native = {key: z[key].copy() for key in z.files}
    with np.load(study / "replay_outputs.npz", allow_pickle=False) as z:
        replay = {key: z[key].copy() for key in z.files}

    checks: list[str] = []
    expected_base = encode(candidates["clauses"][bank["base_proposal_ids"].reshape(-1)]).reshape(
        bank["base_unique_ids"].shape + (41,)
    )
    if not np.array_equal(bank["unique_tokens"][bank["base_unique_ids"]], expected_base):
        raise RuntimeError("base proposal-to-unique-token linkage mismatch")
    checks.append("base proposal ids link exactly to analysed unique-token rows")

    if len(bank["transform_unique_ids"]) != len(bank["transform_rows"]):
        raise RuntimeError("transform row/id length mismatch")
    expected_transformed = []
    for row in bank["transform_rows"]:
        ti, _, _, _, proposal = map(int, row)
        expected_transformed.append(encode_formula(transform_formula(
            candidates["clauses"][proposal], protocol.TRANSPOSITIONS[ti],
        )))
    expected_transformed = np.asarray(expected_transformed, dtype=np.int16)
    if not np.array_equal(bank["unique_tokens"][bank["transform_unique_ids"]], expected_transformed):
        raise RuntimeError("transform row-to-unique-token linkage mismatch")
    checks.append("all transform rows link exactly to independently re-encoded unique-token rows")

    expected_union = {bytes(map(int, row)) for row in np.concatenate([
        expected_base.reshape(-1, 41), expected_transformed,
    ])}
    observed_union = {bytes(map(int, row)) for row in bank["unique_tokens"]}
    if observed_union != expected_union:
        raise RuntimeError("unique token bank has omissions or extras")
    checks.append("unique-token bank is exactly the selected base/transformed union")

    for key, expected in (("unit_ids", protocol.UNITS),
                          ("unit_assignment_ids", protocol.ASSIGNMENT_IDS),
                          ("unique_assignment_ids", protocol.UNIQUE_ASSIGNMENTS)):
        if not np.array_equal(bank[key], expected):
            raise RuntimeError(key + " differs from frozen protocol")
    checks.append("stored unit and assignment inventories equal frozen constants")

    prior_rows = 0
    for path in (protocol.PRIOR_LEDGER, protocol.PRIOR_RENAMING):
        with np.load(path, allow_pickle=False) as z:
            rows = z["tokens"].reshape(-1, 41)
        for row in rows:
            decode(row)
        prior_rows += len(rows)
    checks.append("every historical-ledger row decodes under the frozen grammar")

    if verification["candidate_bank_sha256"] != construction["candidate_bank_sha256"]:
        raise RuntimeError("candidate hash differs between construction and verification")
    if construction["candidate_bank_sha256"] != digest(study / "candidate_bank.npz"):
        raise RuntimeError("candidate artifact hash mismatch")
    checks.append("candidate artifact is hash-linked through construction and verification")

    replay_ids = np.asarray(construction["replay_unique_ids"], dtype=np.int32)
    if not np.array_equal(replay["unique_ids"], replay_ids):
        raise RuntimeError("replay output row ids mismatch")
    if not np.array_equal(native["unique_ids"], np.arange(len(bank["unique_tokens"]), dtype=np.int32)):
        raise RuntimeError("native output row ids mismatch")
    checks.append("native and replay output row identities are exact")

    helpers = protocol.HELPERS_SOURCE.read_text()
    interpretation = protocol.INTERPRETATION_SOURCE.read_text()
    required_helpers = (
        "{f'x{i}': i for i in range(nvars)}",
        "{f'Not(x{i})': nvars+i for i in range(nvars)}",
        "'s': 2*nvars + 3, 'u': 2*nvars + 4",
    )
    if not all(text in helpers for text in required_helpers):
        raise RuntimeError("tokenizer source assertions failed")
    required_interpretation = (
        "np.unpackbits(np.arange(2**config.nvars, dtype=np.uint8)[:,np.newaxis]",
        "[:,-config.nvars:]",
        '"T" if x else "F"',
    )
    if not all(text in interpretation for text in required_interpretation):
        raise RuntimeError("assignment source assertions failed")
    checks.append("token and assignment conventions are directly asserted in bound sources")

    for receipt in (construction, verification, native_receipt, replay_receipt, analysis):
        for key in ("freeze_sha256", "code_sha256", "selfcheck_sha256", "source_sha256"):
            if receipt.get(key) != frozen[key]:
                raise RuntimeError("receipt disagrees with reviewed packet: " + key)
    checks.append("all stage receipts match the externally anchored reviewed packet")

    result = {
        "status": "PASS", "scope": "post-GO linkage and source-convention audit; protected outcomes and gates unchanged",
        "checks": checks, "check_count": len(checks), "historical_rows_decoded": prior_rows,
        "unique_tokens": len(bank["unique_tokens"]), "base_rows": int(bank["base_unique_ids"].size),
        "transform_rows": len(bank["transform_rows"]), "freeze_sha256": frozen["freeze_sha256"],
        "selected_bank_sha256": digest(study / "selected_bank.npz"),
        "analysis_sha256": digest(study / "analysis.json"),
    }
    target = study / "post_go_linkage_audit.json"
    with target.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
