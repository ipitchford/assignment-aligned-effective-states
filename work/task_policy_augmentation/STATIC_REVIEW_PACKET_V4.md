# Exact-input static admission packet v4: cached task-policy screen

Prepared 17 October 2026 before any freeze, research-cache opening, research
fit, threshold calculation, formula generation or native forward. This packet
supersedes v1--v3 only for admission of the current implementation. The earlier
packets and all revision decisions remain audit history.

## Exact review question

Does this implementation faithfully execute the fixed seven-arm cached
comparison, enforce every scientific and resource stopping rule, close the
defects in the two independent and two cross-model revision reviews, and
preserve static isolation?

Return `ADMIT`, `REVISE`, or `REJECT`. If and only if the answer is `ADMIT`, the
signed review must contain exactly one `DECISION:` line, that line must be the
last nonblank line, it must be `DECISION: ADMIT_STATIC`, and the review must
transcribe every code and governance path and digest listed below. These are
machine-checked preconditions for `prepare-freeze`.

## Exact mandatory inputs

| Role | Path | SHA-256 |
|:--|:--|:--|
| Executable | `work/task_policy_augmentation/screen.py` | `262db662f89aa833afc20c7857d89fbde311b667c191e3a447f3f39551b32f30` |
| Static suite | `work/task_policy_augmentation/static_selfcheck.py` | `5513622f82a8ce43b35e4d2f134289cb99d7cc28fbaa5832b72978768dcdc089` |
| Synthetic smoke | `work/task_policy_augmentation/runtime_smoke.py` | `fe806bc507b8ddbe130e41d82fb9d9eab89e02b8f82f85fe3d481a22d132cbe0` |
| Protocol | `work/task_policy_augmentation/protocol.md` | `bf1b032641aa981ed137c5f5e3ea6b3660feb1aad25e4c498789c8d871ef088a` |
| Runtime lock | `work/task_policy_augmentation/requirements-analysis.txt` | `13a45e3554e8dcbf9c2166cb1796280c78ac972a4b085e7dcdf67da08b7dbc85` |
| Static receipt | `work/task_policy_augmentation/static_selfcheck_receipt_v8.json` | `bec2b656b217f296744a79134486e837763fe96cd707c22c8b60d380b059c2e5` |
| Smoke receipt | `work/task_policy_augmentation/runtime_smoke_receipt_v2.json` | `aa8f32f06d55ee7bb116cf379735e83ff64e9c4ff542c9dfb0ecb4bfbf8d1daa` |
| Handoff | `work/task_policy_augmentation/IMPLEMENTATION_HANDOFF.md` | `0c92b4e00dd4c02ca6ea8b2fe687d95a374978e9337dafda4e8dfde874d71e57` |
| Original SAT contract | `work/sat_case/contract.md` | `36a8f2763d4e299d19620db4e58a30e19788ae404f56939225f73cf95c489104` |
| Original pilot implementation | `work/sat_case/fit_pilot.py` | `9884ed49f596d4ca257c38ec160cc8a8faab7633eb6fc6935caa39ef64e56f78` |
| First signed review | `work/next_phase/task_policy_static_admission_independent.md` | `4cf0acd0c29ba9a6ed26c590175284ac22d23585989f1d7fdcecb65b509b5a94` |
| Cross-model review record v1 | `work/next_phase/task_policy_cross_model_static_review_v1.md` | `cfe88d0a8c5bd2123f785bcff86d03b262faf553f2ab5ecb18be62e0ba2c3acb` |
| Cross-model review record v2 | `work/next_phase/task_policy_cross_model_static_review_v2.md` | `c3c0f91561cdb7d78857095a52ba92a2f3fedafba2bbdd7a3d551da679a18447` |
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

1. Recompute every listed mandatory-input digest. Inspect every mandatory input.
   Do not inspect or hash the contents of the three immutable research inputs;
   their identities are already recorded above.
2. Recheck rows/split, SHA slot selection, semantic label indexing, report
   boundaries/fallbacks, action context, learner parity, choices, metrics,
   exact set-valued ceilings, all seven gates and absence of outcome-dependent
   fallback.
3. Verify that the contiguous split is fixed by the original contract and pilot
   implementation rather than inferred from a research-output cache.
4. Verify that allocation and cap checks bracket every fit and prediction and
   extend through exclusive serialization, hashing and the authoritative
   terminal record. Confirm failure receipts retain stage, arm, attempted and
   completed fits, partial metrics/ceilings, model sizes, versions, governance
   and evidence identities, thread environment and final observed resources.
5. Verify exclusive NPZ creation, unambiguous uncommitted `result.json`,
   manifest binding, and crash-atomic exclusive `terminal.json`. Confirm no
   success/error ambiguity.
6. Verify the pre-fit cache contract for token, clause, outcome and native-array
   shapes; Boolean outcome dtype; integer/ranged clauses; and its synthetic
   negative test without opening a research input.
7. Verify that `prepare-freeze` rejects an in-directory, nominal,
   non-admitting, quoted-admission, multi-decision, nonfinal-decision or
   hash-incomplete review; binds all governance and evidence files; and that
   execution rechecks them. Verify no freeze and no run exist.
8. Verify static-suite isolation and all 52 checks. Run it with Python bytecode
   writes disabled and the five numerical thread variables fixed at two. Replay
   to a temporary path and compare byte-for-byte with the v8 receipt.
9. Run the synthetic smoke only on its generated 256-by-84 Gaussian fixture.
   It must produce exactly three trees and fifteen leaves. Replay to a temporary
   path and compare byte-for-byte with the v2 receipt.
10. State exact research-cache, research-fit, research-formula,
    checkpoint-load and native-forward counts for the review.
11. Verify that each finite ceiling emits a complete class-size histogram,
    reachable-member class-size histogram and singleton fraction, and that the
    refinement-monotonicity and near-singleton-saturation caveats correctly
    constrain the claim ladder. Do not require or introduce a new arm in this
    already fixed branch.
12. Verify Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.0;
    the 1,800-second alarm; 4-GiB observed peak-RSS boundary; two-thread
    environment; and 700-tree/10,500-leaf bounds. The documents must describe
    RSS checks as boundary audits, not OS-level allocation prevention.

## Restrictions and output

Do not open research caches or outcome arrays, calculate research thresholds,
fit a research model, generate research formulas, load a checkpoint, freeze,
execute, edit a file or infer likely results. The signed review must list every
inspected path and observed hash, commands run, activity counts, strongest
counterargument, material residual limitations and independence limits. End
with `DECISION: ADMIT_STATIC` only if no material defect remains; otherwise end
with `DECISION: REVISE_STATIC` or `DECISION: REJECT_STATIC`.

Author claim for verification: 52 static checks pass; the exact synthetic smoke
passes; both authoritative receipts replay byte-identically; zero research
activity occurred; no freeze and no run exist. This claim is not admission.

