# Exact-input static admission packet v3: cached task-policy screen

Prepared 17 October 2026 before any freeze, research-cache opening, research
fit, threshold calculation, formula generation or native forward. This packet
supersedes v1 and v2 only for admission of the current implementation; both
earlier packets and the signed v1 `REVISE` decision remain audit history.

## Exact review question

Does this implementation faithfully execute the fixed seven-arm cached
comparison, enforce every scientific and resource stopping rule, close the
three defects in the first signed review, close the bounded governance/cache
defects in the cross-model review, and preserve static isolation?

Return `ADMIT`, `REVISE`, or `REJECT`. If and only if the answer is `ADMIT`, the
signed review must contain the literal line `DECISION: ADMIT_STATIC` and must
transcribe every code and governance path and digest listed below. This is a
machine-checked precondition for `prepare-freeze`.

## Exact mandatory inputs

| Role | Path | SHA-256 |
|:--|:--|:--|
| Executable | `work/task_policy_augmentation/screen.py` | `17e54949336f4db8507c70d954c768f23f27e8d76bfc824801250459b46b50dd` |
| Static suite | `work/task_policy_augmentation/static_selfcheck.py` | `44a4d9c19eba236b92e1a65448562813a9651e738ff7d1c97d9f857357be816f` |
| Synthetic smoke | `work/task_policy_augmentation/runtime_smoke.py` | `4b2d7e39a871b947f1f29ca689c7fa23049fc1d52dfd5169deed2698640a5356` |
| Protocol | `work/task_policy_augmentation/protocol.md` | `4b6c3446896fb8958fa9450f6779678f65d0d871960d3d764bbebd4241bf32fa` |
| Runtime lock | `work/task_policy_augmentation/requirements-analysis.txt` | `13a45e3554e8dcbf9c2166cb1796280c78ac972a4b085e7dcdf67da08b7dbc85` |
| Static receipt | `work/task_policy_augmentation/static_selfcheck_receipt_v7.json` | `c26de3d1117591722ace03c84f3a535b20c8bfd2c0c9760a4c4c3e9ba5a3d423` |
| Smoke receipt | `work/task_policy_augmentation/runtime_smoke_receipt.json` | `674cdaa8abeee05c5a114e051be021708e4653bfa159662afb2b0c75b50c0899` |
| Handoff | `work/task_policy_augmentation/IMPLEMENTATION_HANDOFF.md` | `4254059a264c40a092847fdd490275592633f80b9262a4e6ac69a2b098e85a4c` |
| First signed review | `work/next_phase/task_policy_static_admission_independent.md` | `4cf0acd0c29ba9a6ed26c590175284ac22d23585989f1d7fdcecb65b509b5a94` |
| Cross-model review record | `work/next_phase/task_policy_cross_model_static_review_v1.md` | `cfe88d0a8c5bd2123f785bcff86d03b262faf553f2ab5ecb18be62e0ba2c3acb` |
| Admission card | `work/next_phase/task_policy_augmentation_admission.md` | `be0b514ce6b8dac79beeff7944524d5bf58fe9fec00b867a92606bc59eb0757d` |
| Cache-alignment audit | `work/next_phase/task_policy_cache_alignment_audit.md` | `f7b2c611f6686a98ba923e3cfe1eddd1b6b7ec4454e38543e8c99e3d4e8e9fba` |
| Theory note | `work/next_phase/set_valued_policy_sufficiency.md` | `c167b8a49d58c43e6db9d2f79a008f8704f1a8e74925913d40b60788ba660304` |
| Claim ladder | `work/next_phase/task_policy_claim_ladder.md` | `57342b746af2f63ca012ae2358f908a770c9a174e17a24ea7c31afd73aa4c155` |
| Contribution bridge | `work/next_phase/decisive_contribution_bridge.md` | `3e886d47a0e49b257273e390c1cfb721bdd586bccf6c4ed3290a14b350cec584` |
| Literature boundary | `work/next_phase/task_policy_literature_update.md` | `9e3b1a77ec8d078a927a885f9f0443a88d96895a99add8ac08341bce9815a83a` |

Immutable research inputs are identified, but must not be opened in this review:

| Input | SHA-256 |
|:--|:--|
| `work/sat_case/pilot_bank.npz` | `610d02cc85ba825b7ac3100efd30ad1ec8f90f02e11a9fe8444f0c62a9e2971b` |
| `work/sat_mechanism/original_native.npz` | `54f86e7c03c2366345662ce65a90e62201d71869a60f88e5544a267b49e941c2` |
| `work/sat_mechanism/source_mapping.json` | `5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d` |

## Mandatory checks

1. Recheck rows/split, SHA slot selection, semantic label indexing, report
   boundaries/fallbacks, action context, learner parity, choices, metrics,
   exact set-valued ceilings, all seven gates and no outcome-dependent fallback.
2. Verify that design allocation and cap checks bracket every fit and prediction
   and extend through exclusive serialization, hashing and the authoritative
   terminal record. Confirm failure receipts retain stage, arm, attempted and
   completed fits, partial metrics/ceilings, model sizes, versions, identities
   and final observed resources.
3. Verify NPZ `xb` creation, unambiguous uncommitted `result.json`, manifest
   binding, exclusive `terminal.json`, and no success/error ambiguity.
4. Verify the pre-fit cache contract for token, clause, outcome and native array
   shapes; Boolean outcome dtype; integer/ranged clauses; and its synthetic
   negative test without opening a research input.
5. Verify that `prepare-freeze` rejects an in-directory, nominal, non-admitting,
   or hash-incomplete review; binds the six governance and three evidence files;
   and execution rechecks them. Verify no freeze/run exists.
6. Verify static-suite isolation and all 49 checks. The synthetic smoke may run
   only its generated 256×84 Gaussian fixture and must produce three trees and
   fifteen leaves. State exact research-cache, research-fit, research-formula,
   checkpoint-load and native-forward counts.
7. Verify the refinement-monotonicity and near-singleton-saturation caveats are
   mathematically correct and consistently constrain the claim ladder. Do not
   require or introduce a new arm in this already fixed branch.
8. Verify exact runtime versions, 1,800-second alarm, 4-GiB observed peak-RSS
   boundary, two-thread environment, and 700-tree/10,500-leaf bounds. The docs
   must not describe boundary RSS checks as OS-level prevention.

## Restrictions and output

Do not open research caches/outcomes, calculate research thresholds, fit a
research model, generate research formulas, load a checkpoint, freeze, execute,
or infer likely results. The signed review must list every inspected path and
observed hash, checks and commands run, activity counts, strongest
counterargument, material residual limitations, and independence limits. End
with `DECISION: ADMIT_STATIC` only if no material defect remains; otherwise end
with `DECISION: REVISE_STATIC` or `DECISION: REJECT_STATIC`.

Author claim for verification: 49 static checks pass; synthetic smoke passes;
zero research activity; no freeze and no run. This claim is not admission.
