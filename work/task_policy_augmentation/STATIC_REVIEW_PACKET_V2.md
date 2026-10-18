# Exact-input static rereview packet: cached task-policy screen

Prepared 16 October 2026 after the first independent decision `REVISE` and
before any freeze or research execution. This packet supersedes the earlier
packet only for admission of the revised implementation. The earlier packet
and signed decision remain immutable audit history.

## Review question

Does the revised exact implementation faithfully execute the prespecified
cached comparison, close all three defects in the first review, enforce the
scientific stopping rules, bind commitments before outcome access, and avoid
opening research caches during static verification?

Return `ADMIT`, `REVISE`, or `REJECT`. Any defect affecting rows, labels,
reports, contexts, grouping, metrics, gates, resource accounting, terminal
state, exclusive writes, freeze identity, repeat execution or interpretation is
material.

## Exact files and identities

| Role | Path | SHA-256 |
|:--|:--|:--|
| Executable | `work/task_policy_augmentation/screen.py` | `a127e27134278037c93c174deda6a6b086e20cc8ea3ec6996df048f9e74a77f5` |
| Static suite | `work/task_policy_augmentation/static_selfcheck.py` | `d67cfedfc26b779db303e3c5e96057990fecd293c2c0b8b010de02c6998d4aa6` |
| Synthetic runtime smoke | `work/task_policy_augmentation/runtime_smoke.py` | `4b2d7e39a871b947f1f29ca689c7fa23049fc1d52dfd5169deed2698640a5356` |
| Protocol | `work/task_policy_augmentation/protocol.md` | `82605459730968b1208ed5726e67fe3aebce8cdf7098d307fa6fadf72f24db96` |
| Runtime lock | `work/task_policy_augmentation/requirements-analysis.txt` | `13a45e3554e8dcbf9c2166cb1796280c78ac972a4b085e7dcdf67da08b7dbc85` |
| Static receipt | `work/task_policy_augmentation/static_selfcheck_receipt_v6.json` | `01d19b1754e2890ce38742b5df38c9b494d9f2aaa728dd044c77d40a723f5f28` |
| Runtime-smoke receipt | `work/task_policy_augmentation/runtime_smoke_receipt.json` | `674cdaa8abeee05c5a114e051be021708e4653bfa159662afb2b0c75b50c0899` |
| Implementation handoff | `work/task_policy_augmentation/IMPLEMENTATION_HANDOFF.md` | `c56770f4e3b8aeb93c153be9289d2612c3bc1945a06581716ed6340980bb6dad` |
| First signed review | `work/next_phase/task_policy_static_admission_independent.md` | `4cf0acd0c29ba9a6ed26c590175284ac22d23585989f1d7fdcecb65b509b5a94` |
| Admission card | `work/next_phase/task_policy_augmentation_admission.md` | `be0b514ce6b8dac79beeff7944524d5bf58fe9fec00b867a92606bc59eb0757d` |
| Cache-alignment audit | `work/next_phase/task_policy_cache_alignment_audit.md` | `f7b2c611f6686a98ba923e3cfe1eddd1b6b7ec4454e38543e8c99e3d4e8e9fba` |
| Theory note | `work/next_phase/set_valued_policy_sufficiency.md` | `638796cd8aef283c124b69962601e9784058f1209adcf47f00d2f6613d114e27` |
| Claim ladder | `work/next_phase/task_policy_claim_ladder.md` | `1a07f59eb4cb1f49911579181d9092697498cf8b373effb2aa103730065d1546` |
| Contribution bridge | `work/next_phase/decisive_contribution_bridge.md` | `3e886d47a0e49b257273e390c1cfb721bdd586bccf6c4ed3290a14b350cec584` |
| Literature boundary | `work/next_phase/task_policy_literature_update.md` | `8f6a4f7211585c52e427c4ed245ccbc82c8ef7f7ece39fcc026e981d905f41e0` |

The immutable research-input pins remain:

| Input | SHA-256 |
|:--|:--|
| `work/sat_case/pilot_bank.npz` | `610d02cc85ba825b7ac3100efd30ad1ec8f90f02e11a9fe8444f0c62a9e2971b` |
| `work/sat_mechanism/original_native.npz` | `54f86e7c03c2366345662ce65a90e62201d71869a60f88e5544a267b49e941c2` |
| `work/sat_mechanism/source_mapping.json` | `5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d` |

## Required checks

1. Verify the original packet's checks 1–11 and 14–15 on the revised exact
   inputs: rows/split; hashing; labels; reports; context/cost; learner parity;
   choice/metrics; exact ceiling; seven-part stopping gate; no fallback; freeze
   integrity; static isolation; and interpretation consistency.
2. **First-review R1:** verify a cap check after design allocation and before
   every fit, after every fit, before every prediction, immediately after every
   prediction, after analysis, after each serialization step, and after output
   hashing. Verify final observed elapsed time, RSS, trees and leaves occur in
   the authoritative terminal receipt. Confirm the documentation describes
   boundary auditing rather than OS-level prevention of transient allocation.
3. **First-review R2:** verify attempted/completed fits, current arm/stage,
   available metrics/ceilings, model sizes, versions, input/freeze identities
   and final resource observations survive failure. Verify `result.json` cannot
   be mistaken for authoritative completion and that only `terminal.json`
   commits completion. Test the pure synthetic failure-record fixture without
   research data.
4. **First-review R3:** verify the NPZ is written through an `xb` file handle,
   and every other artifact is also exclusively created.
5. Verify the 1,800-second alarm, 4-GiB peak-RSS cap, two-thread environment,
   700-tree cap, 10,500-leaf cap and exact runtime versions.
6. Verify one run directory, a manifest over the preterminal artifacts, and a
   terminal record binding the manifest hash. Consider failures during every
   output stage, including hashing and terminal creation.
7. Verify the static receipt's 45 checks, zero research-cache opens, zero
   research fits, zero generated formulas, zero native forwards, no freeze and
   no run. The runtime smoke may perform only its generated Gaussian fit.

## Reviewer output and restrictions

State the decision; every material finding; all observed file hashes; whether
the static suite and runtime smoke were rerun; counts of research-cache opens,
research fits, generated research formulas, checkpoint loads and native
forwards; freeze/run state before and after review; strongest counterargument;
and independence limits. End with authorization for `prepare-freeze` only if
the decision is `ADMIT`.

Do not execute `prepare-freeze` or `execute`; inspect research arrays or cached
outcomes; calculate research thresholds; fit a research model; generate
research formulas; load the SAT checkpoint; or infer likely results.

Current author claim: PASS, 45 static checks; synthetic smoke PASS with three
trees and fifteen leaves; zero research activity; no freeze and no run. This is
an input claim for independent verification, not an admission decision.
