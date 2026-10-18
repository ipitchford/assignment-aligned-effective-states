# Independent saved-data audit: SAT binding under renaming

16 October 2026. Model-generated correctness audit, not a external scholarly assessment or human referee report.

**Result: PASS for the saved-data audit. The study's prespecified result remains `CRITERION_NOT_MET`.** No material numerical, denominator, mapping or scope error was found in the reported analysis. Exactly two benchmark conditions fail: P2 primary sensitivity and P2 exact positive-pattern agreement. Neither the audit nor the affine secondary result changes that disposition.

## Exact inputs and audit boundary

| Artifact | SHA-256 |
|---|---|
| Final execution contract | `4b9dc2df3002dbbe56cbcc5d38c119ace801f16e2f3bd018d345deb4eda7af8c` |
| Original implementation freeze | `3a771ecc50d9481c3fceaca79da5a0965e99b77e681ace6a5f628025b2433ced` |
| Selected-data freeze | `041cfd909fd34b75562f794bd1b4bf6aadb1e5e9b67b274ff35d92e4fb2a2d0d` |
| Clerical-recovery freeze | `c93c38d9727e16ea4272a60c73692ee361057a0c2572427dedf1b0674f33a61a` |
| Recovery `MANIFEST.json` | `dd02ec8bb5ed5ee79c04cbdb3eed302315ad112452df1151444d15f328b760a3` |
| `selected_bank.npz` | `4c4e668f80328ed34349448c1506c0702bce0a35464c0896a9b9181d9d2adf8a` |
| `native_outputs.npz` | `cf0f69826a5fc788415809af3382eff01d40ce891c411d80e99ec886e05316b2` |
| `replay_outputs.npz` | `67d3b8282e95c5ca4e3822cc7b4054419abc34cfd579c4834f9fc030a4daf232` |
| `analysis.json` | `28fe7436ec4aea45e930281390d8a9fa542c04be13fbde8c597da2e9210b41bb` |
| `bounded_metrics.npz` | `10e60a56d151a7440668070297541c27ff927cc8c0d05580579ff8d6b4ef92ae` |
| `bootstrap_indices.npz` | `8e72d2b3b4fea0436a8fed0ea390bfe08e9c15f948d8eb0e14510fb84da7139b` |

Study paths are under `work/sat_renaming_launch_recovery/`; the original failed attempt remains under `work/sat_renaming/`. Audit code and receipts are in the new `work/sat_renaming_independent_audit/` directory. Every inspected input has a byte count and hash in `receipt.json`. All inspected files remained unchanged.

The final contract, recovery admission, frozen scientific implementation and saved outputs were inspected. The reviewer authored the ordinary-loop Boolean oracle and bank verifier before construction, independently of the vectorized sampler. The subsequent arithmetic audit was written separately and does not import `analyze.py`, `protocol.py`, the sampler, a model runtime or Torch. This is code-level independence within the same model-assisted research programme, not a blind or independently executed neural replication.

No new formulas, model loads, neural forwards, fits or scientific resampling were performed. The 2,000 saved bootstrap draws were reused. One explicitly identified **computational replay** regenerated the already-declared seed's 2,000 index blocks and found exact equality; these are not additional study draws or a replacement sample.

## Semantic bank and execution provenance

The separately executed frozen `verify_bank.py` returned PASS. Its Boolean loops rederived all 16,384 endpoint/valuation counts, all 2,048 four-unit semantic labels and all 512 task labels. It checked both cohorts, the unique satisfying assignments, complete UNSAT endpoints, the common five-field action descriptor, retained literal and new-variable conditions, clause/slot preservation, renaming involution, all count updates, action commutation and literal-position preservation.

There are 32 distinct descriptors for each pair, 128 distinct origin/renaming orbits and 512 distinct endpoint token sequences. The complete six-entry ledger was independently reconstructed from its five pinned source files: 9,446 source rows collapse to 8,957 distinct prior inputs. Every selected endpoint and its renamed form are absent from that ledger. The original encoder, source mapping and final contract identities match their preconstruction pins. Freshness is consequently established relative to the known campaign ledger only.

All implementation-freeze and selected-data-freeze identities, both original and recovery final inventories, and the recovery's byte-identical copies passed. The original attempt records a missing-Torch import failure before `main`, no native-start artifact and zero attempted/completed native charges. Its failed status remains intact. The recovery preserves the original native deadline and completes within it. Journals independently account for 64 initial and two replay eight-row batches: **512+16 input forwards, 66 batch calls, zero fits**. No proposals were regenerated during recovery.

Every saved individual batch agrees with the combined arrays. All retained native arrays are finite float32 with the expected dimensions; selected postactivations equal exact ReLU of the selected preactivations and match their coordinates in the complete postactivation array. The 16 saved replay inputs, in the same batch context, reproduce every retained numeric array exactly. Saved distances to 0.5 and full-vocabulary winning gaps also recompute exactly.

## Primary criteria, mappings and task results

| Quantity | P1: 261 ↔ 291 | P2: 411 ↔ 492 |
|---|---:|---:|
| Primary true positives | 63/64 | 56/64 |
| Primary true negatives | 448/448 | 448/448 |
| Complete positive patterns | 31/32 | 24/32 |
| Off-pair false positives | 0/512 | 0/512 |
| Original-direction true positives | 31/32 | 24/32 |
| Renamed-direction true positives | 32/32 | 32/32 |
| Positive-edited identity-map advantage | 0.484375 | 0.375000 |
| Positive-edited wrong-map advantage | 0.484375 | 0.375000 |

The fixed cutoffs require at least 58/64 primary true positives and 26/32 positive patterns per pair. P2 fails both; all other benchmark conditions pass. P2's **original** direction fails the separately fixed 29/32 rule, while its renamed direction passes. It therefore does not meet the contract's definition of a localized renaming failure, which requires the original direction to pass and renamed direction to fail.

There are nine erroneous primary events: one original edited endpoint for unit 261 and eight for unit 411. All are false negatives; their raw values and row identities are retained in `unit_event_errors.json`. Seven of those nine endpoints nevertheless receive the correct native task answer. No failed task or unit response was removed.

The correct mapping `(261↔291, 411↔492)`, identity mapping and wrong mapping `(261↔492, 291↔411)` were recomputed on the same four-unit vectors. Full-origin and positive-edited errors, both mapping contrasts, all signed changes and primary-unit transport/change counts match the saved arrays. The equally pooled interval is **[0.390625, 0.468750]** for each positive-edited mapping advantage. This positive conditional mapping contrast coexists with the failed source-extension criterion; it does not override it.

Full 16-token argmax scoring gives **505/512** correct task answers: **126/128 SAT** and **379/384 UNSAT**, with no non-answer argmax. The corresponding pooled balanced accuracy is **0.9856770833**. Cohort, quartet and joint pattern/task summaries match the saved records. Negative-control task sensitivity remains undefined because that cohort has no SAT labels. The 2,048-row reader-facing CSV preserves every event and its associated task outcome.

## Numerical sensitivity and bootstrap replay

The audit independently formed each postactivation's admissible bit set using the contract's `0.0001+0.00001*abs(value)` radius. It also formed possible classes by comparing each logit's upper endpoint with the lower endpoints of the other 15 classes. **There are zero ambiguous native bit positions and zero ambiguous native task classes.** Thus each joint endpoint or signed-change possibility set has one member on this bank. All current mapping, exact-pattern, direction, class and bootstrap bounds must—and do—collapse to their nominal values. The failed criterion is robust under the declared heuristic intervals.

All 47 block metrics and their fixed denominators were reimplemented with explicit event loops. Every point estimate, all 2,000 stratified bootstrap outcomes, all nominal/lower/upper columns and every saved JSON interval agree. The audit used saved index multiplicities to recompute resampling means and independently interpolated sorted values at the linear 2.5%/97.5% quantiles. No draw was dropped or replenished. All 14 benchmark conditions, their integer cutoffs, strict mapping inequalities and directional classifications reproduce.

These checks verify the interval arithmetic actually used by this bank. The data contain no ambiguous decisions, so they do not empirically exercise the implementation's general multi-possibility branches. The stipulated radii remain engineering sensitivity conventions, not proven floating-point error bounds or confidence intervals for exact-real outputs.

## Secondary affine accounting

The reconstruction was independently associated as `(resid_mid + b_out + full_post W_out^T) W_U`. All 8,192 native logits satisfy the declared parity tolerance; the largest residual is approximately **1.347612×10⁻⁵**. Independently calculated reconstruction and clamp values differ from the study's float64 arrays by at most **3.56×10⁻¹⁴** across all numerical audit comparisons.

For every edited endpoint, the audit recalculated both fixed pair clamps, their signed SAT-minus-UNSAT contributions, all 16 resulting logits, argmaxes, possible-class sets, winning gaps, correctness and stable-change indicators. The nominated-pair clamp changes **115/128 positive edited decisions**, all from correct to wrong and all stable under the stipulated heuristic intervals. The other fixed pair changes **0/128** positive decisions; neither pair changes a negative-control decision. All reported contribution means agree.

This remains a defined downstream affine clamp with edited residual and all other postactivations held fixed. The audit does not convert it into a patched-native float32 experiment, a natural intervention, upstream mediation or uniquely necessary semantic routing. It supplies no role for clipped negative preactivations in the present readout.

## Findings and limits

No material implementation or reporting defect was detected within this saved-data scope. `receipt.json` records 5,943 successful identity/arithmetic/export checks; the number is bookkeeping, not an evidential-quality score. The strongest scientific counterargument remains that assignment-selective scalar margins or confidence-plus-syntax mechanisms can produce this pattern and clamp behavior. The study measures conditional validity of two published labels under its selected construction; it does not identify a unique mechanism.

The selected descriptors form a deliberately constructed finite cohort, not an independent random sample of units or formulas. Matching, the shared proposal pool and global exclusions limit bootstrap interpretation. The follow-on question was chosen after earlier development findings. This audit did not independently regenerate the 100,000-proposal pool or replay every hash-based selection decision, and it did not execute the checkpoint. Source approximation guarantees, training-set novelty, broader generalisation, causal uniqueness and broader significance remain outside its scope.

Audit artifacts:

- `work/sat_renaming_independent_audit/recompute.py`: SHA-256 `1cc39749307d49d80a1ccc31c4a99c187c5551e1f043f8a87eb689881fb7c90e`.
- `work/sat_renaming_independent_audit/receipt.json`: SHA-256 `fd281975839f8f059b5457faeec15c5bb10f8bbdcdeceae91a6334ed670d4ff0`.
- `work/sat_renaming_independent_audit/bank_replay.json`: SHA-256 `4a413939c642d5a8d4797446e13dafac8d03ba873968ff47745abe83f1528b38`.

The audit scripts preserve exclusive-write behavior. Reproduction should use a fresh output directory or the separately labelled portable packaging wrapper; it must not overwrite these receipts or invoke a scientific driver.
