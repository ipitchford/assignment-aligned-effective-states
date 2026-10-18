# Draft: source-fixed SAT unit binding under one variable renaming

16 October 2026. **Design only: no new formulas, checkpoint loads, native forwards, fits or bootstrap draws have been produced for this proposal.** The next possible action is a bounded, truth-only construction feasibility check after review and an implementation freeze. This draft is not an instruction to start extraction.

The question is whether the published assignment-to-unit interpretation predicts selective changes in *which native unit responds* when a single-clause repair is transported through `π: x1 ↔ x2`. This is a separate hypothesis under the existing programme. The stopped continuous-margin comparison remains stopped; no decoder, threshold or baseline from that branch will be tuned.

## 1. Source scope and fixed objects

Read alongside `sat_invariance_admission.md`, SHA-256 `4e5447713dffffb9c0379b632a406b183b403c25943667f4fdd35ec38fb896d2`. [Palumbo et al. (ICML 2025), Table 2 and Appendix D.2](https://arxiv.org/html/2407.13594v2), supplies approximate assignment predicates for thresholded activation and already derives a weighted clause-count mechanism. Its interpretation has distribution-dependent errors and documented satisfying-assignment misses. It does not guarantee this selected cohort, equal raw margins, or symmetry of every hidden unit. This experiment would test a restricted extension of that interpretation.

Variable indices are **zero-based**: assignments list `x0,x1,x2,x3,x4`. Thus π swaps the second and third characters, not the first two. Freeze these anchor orientations and no others:

| Pair | Anchor unit i, assignment a | Partner unit j, assignment b=πa |
|---|---|---|
| P1 | 261, TTFTT | 291, TFTTT |
| P2 | 411, TTFFT | 492, TFTFT |

All four units are unique singleton rules. Neither duplicate rules nor the other thirteen source-nominated units enter this study. Use the source-code boundary `B_u(G)=1[post_u(G)>=0.5]` at the final token of the final MLP. Store raw preactivations and postactivations, but do not test raw-value equality or select examples using either.

Each selected origin is observed together with its renamed copy. That quartet tests both directions of transport; reversing it does not create another independent origin or another sample quota.

## 2. Grammar, edit and exact semantic eligibility

Use the native ten-clause, five-variable grammar, allowing repeated clause occurrences and retaining all ten ordered slots. A clause has two different variables, and its literals are sorted by **variable index**, not signed-token value. Exclude the unordered variable pair `{x1,x2}` in every clause of every endpoint. This leaves 36 of the original 40 signed clauses. Because the swapped indices are adjacent, this restriction makes π preserve literal positions as well as clause slots. Nevertheless, implement renaming and canonical sorting explicitly and assert that no literal position changed.

For assignment s, let `q_s(G)` count falsified clause **occurrences**, and let `M(G)` be the set of satisfying assignments among all 32 truth assignments. Fix an action descriptor `d=(k,c,ℓ,v,c_new)`: slot k, old canonical clause c, kept literal ℓ, new variable v outside c, and new canonical clause. Require `ℓ(a)=0`, `ℓ(b)=1`. The other literal of c is false under a. Replace it with the literal of v that is true under a. Require the old and new clause to belong to the restricted grammar. The kept literal makes both clauses true under b, so `q_b` is unchanged.

Two truth-only cohorts share the same action descriptors:

- **Positive repair F:** `M(F)=∅`, `q_a(F)=1`, `q_b(F)>=2`; k is a's unique falsified occurrence; and explicitly `M(F′)={a}` after replacement. Uniqueness is a separate oracle check, not a consequence of `q_a=1`.
- **Negative control H:** `M(H)=∅`, `q_a(H)>=2`, `q_b(H)>=2`, H has exactly the same old clause c at k, and explicitly `M(H′)=∅` after the same replacement. The count condition proves only that a remains false; the complete UNSAT endpoint must also be checked.

For each origin G, freeze the four concrete inputs `(G,G′,πG,πG′)`. Exact truth-table checks must verify renaming involution, `q_{πs}(πG)=q_s(G)` for every s, `π(E_d(G))=E_{πd}(πG)`, count decrements, and all endpoint solution sets. The oracle labels are `(UNSAT,SAT,UNSAT,SAT)` for F and four UNSAT labels for H. The renamed positive endpoint has unique solution b. The other pair's two assignments are false at every endpoint in both cohorts.

## 3. Bounded construction and selection before any native extraction

Freeze a complete sampler, encoder, independent Boolean oracle, selection routine, analysis routine, environment manifest and this protocol before construction. The proposed numerical choices are fixed as follows:

- One candidate stream: NumPy `PCG64(2026101604)`, at most **100,000** independent ordered ten-clause draws, each clause uniform from the explicitly saved ordered list of 36 allowed clauses. No adaptive reseeding, second pool or quota relaxation. One CPU thread and a 20-minute wall cap; a cap hit is a feasibility stop, not proof that the cohort is empty.
- Inspect both fixed anchor pairs on the same stream. Record trial indices, truth counts, complete solution sets, eligibility counts and exclusion reasons. Retain at most **one eligible action per origin and pair**, selected by the smallest domain-separated SHA-256 of `(pair, canonical renaming-orbit bytes, complete action descriptor)`. This action choice precedes descriptor matching and never reads neural outcomes. Save the number of eligible actions as a diagnostic; origins with more possible edits must not contribute more rows.
- Within each pair, group the one-action eligible origins by complete action descriptor and cohort. Sort descriptors and candidate origins by separately labelled SHA-256 keys. Select **32 distinct descriptors per pair**, each with one positive and one negative origin. Select in fixed P1/P2 round-robin order; within a descriptor choose the first positive origin and then the first compatible negative origin in hash order. Skip a descriptor if that first positive has no compatible negative; do not search alternative edits or responsive examples. Hash encodings and all tie-breaks must be specified byte-for-byte in the implementation freeze.
- Globally forbid repeated origin/renaming orbits and repeated endpoint token sequences among selected blocks, including across pairs and cohorts. The canonical orbit identifier is the lexicographically smaller byte sequence of `(G,πG)`; this identifies an orbit without changing its frozen anchor orientation. A formula generated twice is one candidate origin, retaining its first trial occurrence. If two matching candidates share any of their four endpoint inputs, they are incompatible.
- Exclude any selected endpoint or its renamed form present in the known prior native-input ledger. At minimum this ledger is derived from `sat_case/pilot_bank.npz:tokens` and `sat_mechanism/query_tokens.npz:unique_tokens`; freeze its full input-file inventory and hashes. This makes “fresh” mean absent from the known campaign inputs. It does **not** establish absence from checkpoint training data.

There are **64 matched blocks**, each containing two distinct origin orbits: 128 origin orbits total, 64 positive and 64 negative. Insufficient support for either pair, any unresolved identity collision, oracle disagreement, or inability to meet the fixed runtime bound stops before checkpoint loading. Report available descriptors, origins, slot distributions, count distributions, action multiplicities and every exclusion. Do not replace unique-SAT eligibility, add another permutation, weaken the forbidden-variable-pair rule, select different units, or extend the pool.

A successful construction is only a feasibility result. Freeze the selected bank, all 512 ordered endpoint rows, complete semantic labels, matched-block identities, replay indices and file hashes before native extraction. No activation, native class, confidence, clipping or threshold distance may enter selection. The second independent oracle should evaluate literal truth with ordinary Boolean loops rather than reusing the sampler's vectorized count implementation. This oracle-assisted cohort is not a report-only deployment policy.

## 4. Native extraction and cost ceiling

The proposed native ceiling is **512 initial input forwards plus 16 fixed replay forwards = 528 input forwards**. A batched call still costs its number of input rows. Use the already pinned unmodified CPU float32 checkpoint, two CPU threads, batch size 128, no gradients and no fitting. The extraction/replay phase has a ten-minute wall cap; partial results are preserved and marked incomplete rather than used to relax the design.

Capture all four nominated units, all 16 output logits and the metadata needed for the affine check below. Do not run a search over other units. Task prediction is the full 16-token argmax; SAT is token 13 and UNSAT token 14. Any other token counts as an incorrect task prediction and is reported separately. Do not silently renormalize to the two answer tokens.

The 16 replay rows are all eight inputs in the first frozen matched block of each pair. Replay once in a separate process with the same pinned runtime. Require finite arrays, exact ReLU consistency, `allclose(atol=1e-4, rtol=1e-5)` for retained numeric arrays, identical `>=0.5` events and identical full-vocabulary argmax labels. Save distances to 0.5 and the winning-logit gaps for every input, not just failures. Retain threshold-near rows; do not replace or selectively rerun them. A failed replay stops scientific adjudication. It is a numerical replay, not exact-real verification.

For sensitivity reporting define a numerical interval around each saved scalar with radius `1e-4+1e-5*abs(value)`. Report which bit/class decisions could change inside those intervals and conservative best/worst metric bounds; do not claim those intervals are proven roundoff bounds. No label may be changed merely to improve an outcome.

Record original and replay input counts, batch calls, wall/CPU time, peak resident memory, runtime versions, checkpoint/source hashes, retained data bytes, analysis time and bootstrap bytes. No input forwards are reserved for an adaptive pilot. There are zero learner fits and zero threshold searches. Analysis uses at most one CPU thread, 2,000 bootstrap draws and a five-minute wall cap; an incomplete analysis is reported without extra draws.

## 5. Predictions, controls and denominators

The primary two-unit prediction for every positive origin is:

| Bit | G | G′ | πG | πG′ |
|---|---:|---:|---:|---:|
| B_i | 0 | 1 | 0 | 0 |
| B_j | 0 | 0 | 0 | 1 |

All eight primary bits are predicted zero in negative controls. The other pair's eight bits are predicted zero for every origin. Also report signed response changes `Δ_u(G)=B_u(G′)-B_u(G)`; negative changes remain errors rather than being clipped away.

With `N+=N−=32` **per pair**, retain the following separate denominators:

| Quantity | Per pair | Both pairs |
|---|---:|---:|
| Independent matched blocks | 32 | 64 |
| Selected origin orbits | 64 | 128 |
| Primary expected-positive bits | 64 | 128 |
| Primary expected-negative bits | 448 | 896 |
| Off-pair expected-negative bits | 512 | 1,024 |
| Primary two-unit endpoint transport comparisons | 256 | 512 |
| Four-unit endpoint comparisons per fixed mapping | 512 | 1,024 |
| Native task classifications | 256 | 512 |

Report primary sensitivity and specificity separately, their balanced mean, exact eight-bit pattern agreement separately for positive and negative origins, off-pair false-positive rate, and the original/renamed direction-specific results. Macro-average the two pairs equally; do not present four units as an independently sampled population. Positive-origin complete pattern agreement has denominator 32 per pair, 64 total. Report it jointly with complete four-input task correctness, without filtering out task failures from the unit analysis.

The task labels comprise 128 SAT and 384 UNSAT endpoint rows. An always-UNSAT predictor therefore achieves 75% ordinary accuracy. Report task sensitivity, specificity and balanced accuracy, positive/control cohorts, and complete-quartet accuracy. Task correctness is a distinct result, not an eligibility filter.

For transport controls define the vector `V(G)=(B_261,B_291,B_411,B_492)` and three frozen mappings: correct `P=(261↔291,411↔492)`, identity I, and wrong `W=(261↔492,291↔411)`. For each origin and each mapping M, compute

`e_M(G) = [Hamming(V(πG),M V(G)) + Hamming(V(πG′),M V(G′))]/8`.

All mappings use the same four-unit vector, inputs and denominator. Separately score the positive edited endpoints only, `e_M^+(G)=Hamming(V(πG′),M V(G′))/4`, and the four-unit signed-change vectors. The decisive mapping contrasts are `e_I^+−e_P^+` and `e_W^+−e_P^+`, paired on the same positive origins. Do not manufacture an advantage by pooling all-zero negatives. These conditional transport comparisons use the observed unrenamed response; they are not predictions of an unseen edit from its prestate. The source-label pattern table is separately frozen before outcomes.

The matched negative cohort controls the exact token/action descriptor in a nonrepairable context. Wrong-member and off-pair responses control indiscriminate unit rises; the identity mapping controls persistence of physical unit identity. Equal semantic task labels do not equalize native confidence. A positive result could reject simple assignment-blind global-rise or fixed-unit accounts on these inputs, not every confidence-plus-syntax model or every semantic scalar-margin implementation. The main design includes no new readiness forecast or continuous comparator.

## 6. Uncertainty and proposed interpretation rules

Use **2,000** prespecified `PCG64(2026101605)` bootstrap draws, stratified by the two pairs. Draw 32 matched-block indices with replacement within each pair and carry both origins, their four endpoints, all four units and all comparator outcomes together. This pairs the comparisons and retains the deliberate matching dependence. Store the complete draw-index array and all statistics. Use the 2.5% and 97.5% empirical quantiles with an explicitly frozen quantile convention. No draw is replenished; missing required statistics propagate to an unresolved result.

Intervals describe variability among the selected origin blocks conditional on the two named pairs, construction rules and one checkpoint. They do not create a random sample of units or establish a universal source guarantee. With 32 blocks per pair, rare failures are poorly resolved; an all-success bootstrap can be degenerate and must not be described as zero population error. Do not select a new sample size from interval width.

Proposed finite-cohort diagnostic benchmark, to be settled by the design review **before** the implementation/selection freeze: each pair must have primary sensitivity and specificity at least 0.90, positive exact-pattern agreement at least 0.80, and off-pair false-positive rate at most 0.10. In addition, the two pooled positive-edited mapping contrasts must each have a strictly positive lower bootstrap endpoint. These are author-chosen stress-test tolerances, not thresholds justified by the original paper; they support only a finite-cohort statement. Report every measure even if one condition fails. Do not optimize these tolerances after construction or extraction.

Classify outcomes transparently:

- **Selective transport supported on this cohort:** the diagnostic benchmark is met; identify separately whether task repair also succeeds. Stronger evidence is original selectivity plus preserved renamed selectivity and a clear advantage over both fixed mappings.
- **Source-extension failure:** inadequate nominated-unit selectivity, indiscriminate rises, or wrong/unchanged mapping performing as well exposes a failure of the proposed extension. If unrenamed selectivity is good but renamed selectivity fails, the defect is more specifically about transport. If both fail, the selected stress domain may be outside the source approximation's useful support.
- **Task/representation dissociation:** correct task classifications with failed nominated bits indicate that the network can solve these inputs without this small reported interpretation succeeding; correct bits with failed classifications show those bits do not ensure the task decision. Neither outcome is repaired by post hoc unit relabeling.
- **Unresolved:** missing semantic support, runtime/replay problems or numerically ambiguous benchmark status. Preserve them without replacing cases or moving the cutoff.

This is one bounded source-fidelity study, not a multiple-direction search until significance. No success automatically establishes a general abstraction, a superior decision policy, broad causal routing, originality relative to Palumbo, or a broader generality.

## 7. Separately labelled affine output accounting

Static inspection of the pinned `model.py` and CPU loader shows a final MLP residual addition followed directly by linear unembedding, with no final normalization. At the final token, writing `h=post`, `r=final resid_mid`, and `C=W_out^T W_U`, the real-arithmetic identity is

`z = r W_U + b_out W_U + h C`.

This source-level check is complete; numerical checkpoint parity has **not** been run for this proposal. During the admitted initial forwards, cache the final-position 512-vector h and 128-vector r solely for full-logit reconstruction, and save the fixed matrices needed for that check. Freeze a separate checker that verifies all 16 reconstructed logits on every endpoint against the native output with the stated tolerance. No hidden-unit search may use the additional cached coordinates. If this parity check fails, omit the secondary clamp interpretation and report the failure; do not alter the model or add forwards.

For each edited endpoint, a simultaneous clamp of the nominated pair Q to the corresponding unedited values, keeping edited r and every other postactivation fixed, has affine logits

`z_clamp = z_edit − (h_edit,Q − h_origin,Q) C_Q`.

Use the matching renamed origin when clamping a renamed edit. Compute the same quantity for the other fixed pair as a control, on all positive and negative endpoints. Retain signed SAT-minus-UNSAT contributions and full 16-way argmax changes, including zero, harmful and non-answer outcomes. Do not choose units or cases because their coefficient is large. Use a frozen float64 arithmetic replay from the saved float32 arrays and preserve reconstruction discrepancies. This requires no extra native input forward and does not justify a claim of bitwise equivalence to a patched float32 rerun.

This is routine affine contribution accounting and an explicitly defined downstream activation clamp, potentially off-manifold. It may show how much these pair-specific post changes contribute to the final logit decision while everything else is held fixed. It neither discovers the readout mechanism nor establishes upstream mediation, necessity under natural interventions, or a role for clipped negative preactivations in the present output. Little clamp effect does not by itself refute semantic binding, because other units and the residual stream may carry redundant evidence.

## 8. Provenance and contribution boundary

Pinned inputs inspected for this draft:

| Input | SHA-256 |
|---|---|
| `sat_case/upstream/model.py` | `7555e1f2f27546225397aa12de4fde1f6c903e9804766de477fe4c02770691ea` |
| `sat_case/runtime.py` | `bbc76d5f2d4a856944f54269a6da2d388f0ea6aee81ab03d9486e2e5c8bca230` |
| `sat_mechanism/source_mapping.json` | `5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d` |
| `next_phase/sat_semantic_theory.md` | `e9c4f571ada0a9568b5fd0e455f2f2696869ce97739e20d058420dd3b1400566` |

The loader pins checkpoint SHA-256 `d587446c64b7a8d68de93808ccf84a84364b85dcaf2149b8983b77895e24046d`; no checkpoint was loaded to write this draft. A future execution freeze must bind the actual file and complete implementation, rather than rely on this textual record.

The nearest prior remains Palumbo's own interpretation. Variable-renaming invariance is established in the design of [NeuroSAT, §3](https://arxiv.org/pdf/1802.03685v4); renaming is also an existing logical-reasoning stress test in [Es-sebbani et al., §3.5 and Appendix F](https://arxiv.org/html/2602.12665v1). Intervention correspondence has a mature formal treatment in [Geiger et al., JMLR 2025](https://jmlr.org/papers/v26/23-0058.html). A successful result here would supply new measured scope for two particular source-fixed bindings under controlled consequential edits. A negative result would delimit that scope. Neither outcome, by itself, closes the larger paper's significance gap.
