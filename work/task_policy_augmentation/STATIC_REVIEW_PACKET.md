# Exact-input static review packet: cached task-policy screen

Prepared 16 October 2026 before any freeze or research execution. This packet
defines the complete input to an independent static admission review. It is not
an admission decision and contains no result.

## Review question

Does the exact implementation faithfully execute the prespecified cached
development comparison, enforce its scientific stopping rules, bind all
computational and inferential commitments before outcome access, and avoid
opening research caches during static verification?

The reviewer should return `ADMIT`, `REVISE`, or `REJECT`. `ADMIT` requires no
unresolved material defect. A cosmetic comment may accompany admission. Any
defect affecting rows, labels, reports, contexts, grouping, metrics, gates,
resource limits, freeze identity, repeat execution or result interpretation is
material.

## Exact files and identities

| Role | Path | SHA-256 |
|:--|:--|:--|
| Executable | `work/task_policy_augmentation/screen.py` | `be9d6c5a3389bcd7b12837c78d3446e78241ad67c41844a079d5ee941878bf9d` |
| Static suite | `work/task_policy_augmentation/static_selfcheck.py` | `26d0fce74f75d227a2952e8b532b2adbddc6acb524d9faf4cbdb18fef9178b47` |
| Synthetic runtime smoke | `work/task_policy_augmentation/runtime_smoke.py` | `4b2d7e39a871b947f1f29ca689c7fa23049fc1d52dfd5169deed2698640a5356` |
| Protocol | `work/task_policy_augmentation/protocol.md` | `2cb31cda236a7a283e31846fb715820a3f3554ad3b73e5bd325e6d8822809f29` |
| Runtime lock | `work/task_policy_augmentation/requirements-analysis.txt` | `13a45e3554e8dcbf9c2166cb1796280c78ac972a4b085e7dcdf67da08b7dbc85` |
| Authoritative static receipt | `work/task_policy_augmentation/static_selfcheck_receipt_v5.json` | `8e8d6bc62992857bbe3e6ad20bb729293327c8ca0c563d260c5c1dc4736ebdbe` |
| Runtime-smoke receipt | `work/task_policy_augmentation/runtime_smoke_receipt.json` | `674cdaa8abeee05c5a114e051be021708e4653bfa159662afb2b0c75b50c0899` |
| Implementation handoff | `work/task_policy_augmentation/IMPLEMENTATION_HANDOFF.md` | `652b6b403c6c66250a6b1e1c339b5add11fe635b5fc44a8f01d39feb393c9b15` |
| Admission card | `work/next_phase/task_policy_augmentation_admission.md` | `be0b514ce6b8dac79beeff7944524d5bf58fe9fec00b867a92606bc59eb0757d` |
| Cache-alignment audit | `work/next_phase/task_policy_cache_alignment_audit.md` | `f7b2c611f6686a98ba923e3cfe1eddd1b6b7ec4454e38543e8c99e3d4e8e9fba` |
| Finite-ceiling theory note | `work/next_phase/set_valued_policy_sufficiency.md` | `638796cd8aef283c124b69962601e9784058f1209adcf47f00d2f6613d114e27` |
| Prospective claim ladder | `work/next_phase/task_policy_claim_ladder.md` | `1a07f59eb4cb1f49911579181d9092697498cf8b373effb2aa103730065d1546` |
| Contribution bridge | `work/next_phase/decisive_contribution_bridge.md` | `3e886d47a0e49b257273e390c1cfb721bdd586bccf6c4ed3290a14b350cec584` |
| Literature boundary | `work/next_phase/task_policy_literature_update.md` | `8f6a4f7211585c52e427c4ed245ccbc82c8ef7f7ece39fcc026e981d905f41e0` |

The executable binds these immutable research inputs without requiring a static
reviewer to read their arrays:

| Input | SHA-256 |
|:--|:--|
| `work/sat_case/pilot_bank.npz` | `610d02cc85ba825b7ac3100efd30ad1ec8f90f02e11a9fe8444f0c62a9e2971b` |
| `work/sat_mechanism/original_native.npz` | `54f86e7c03c2366345662ce65a90e62201d71869a60f88e5544a267b49e941c2` |
| `work/sat_mechanism/source_mapping.json` | `5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d` |

## Required review checks

1. **Rows and split.** Verify 3,072 fit origins and 1,024 assessment origins,
   one SHA-selected slot per origin, 39 ordered alternative clauses per menu,
   and no origin crossing the fixed split.
2. **Hash selection.** Verify the study tag, canonical little-endian signed
   64-bit token bytes, first eight SHA-256 digest bytes interpreted big-endian,
   and modulo-ten slot selection.
3. **Semantic labels.** Verify that the target is post-edit UNSAT for the exact
   origin, slot and candidate clause, and that every semantic outcome remains in
   the evaluation denominator.
4. **Reports.** Verify fit-only lower medians, strict negative-bin boundaries,
   four-bin pre/post definitions, empty-bin fallbacks, float32 retention,
   68-column padding, and seven fixed arms.
5. **Context and cost.** Verify that each decoder receives only its pre-edit
   report plus the 4/6/6-bit slot/old/new descriptor. The report must be computed
   before action context is appended.
6. **Learner parity.** Verify identical histogram-gradient-boosting settings,
   supervision, rows, feature width, seed and budget for all learned arms.
7. **Choice and metrics.** Verify clause-ID tie breaking; all-menu success;
   reachability-conditioned success; regret; fixed coverage thresholds; and the
   stated +1/-4/0 abstention utility.
8. **Exact ceiling.** Verify grouping by exact report bytes, slot and old clause;
   set-valued success accounting; classwise best action; total/mean regret; and
   no claim beyond the finite assessment bank.
9. **Stopping gate.** Verify all seven Boolean requirements, including five-point
   learned and exact-ceiling gains over published bits and matched post-ReLU,
   learned noninferiority to matched pre-ReLU, coverage and positive utility.
10. **No result-dependent fallback.** Verify that failure cannot promote another
    arm, learner, threshold, seed, split or outcome subset.
11. **Freeze integrity.** Verify that preparation binds code, runtime lock,
    immutable inputs, the independent admission and all five governance files;
    execution rechecks each identity and refuses absent or repeated freezes.
12. **Resource enforcement.** Verify the 1,800-second alarm, 4-GiB peak-RSS cap,
    two-thread environment, 700-tree cap, 10,500-leaf cap and exact runtime
    versions before research fitting.
13. **Output integrity.** Verify exclusive writes, one run directory, complete
    metrics/ceilings/gates/model sizes/resource receipt, and retained error
    record on failure.
14. **Static isolation.** Inspect the static suite and receipt to confirm zero
    research-cache opens, fits, generated formulas, native forwards, freezes and
    runs. The one runtime smoke fit must use generated Gaussian data only.
15. **Interpretation consistency.** Verify that protocol, admission, claim ladder
    and contribution bridge agree that a learned gain without a material ceiling
    gain stops the branch and cannot be called an information repair.

## Reviewer output requirements

The admission document must state:

- decision and material findings;
- every inspected path and observed SHA-256;
- whether the reviewer executed the static suite and synthetic runtime smoke;
- explicit counts of research-cache opens, research fits, generated research
  formulas, model/checkpoint loads and native forwards;
- whether any freeze or run artifact existed before or after review;
- the strongest remaining counterargument;
- independence limits, including any shared model/system context;
- a final sentence authorizing `prepare-freeze` only when the decision is
  `ADMIT`.

The reviewer must not execute `prepare-freeze` or `execute`, inspect cached
research outcomes, calculate thresholds from research arrays, fit a research
model, generate formulas, load the SAT checkpoint, or infer likely results.

## Current pre-review state

Authoritative static receipt: PASS, 40 checks, zero research fits, zero native
forwards, no freeze and no run. This statement is an input claim for the
reviewer to verify, not a substitute for independent admission.
