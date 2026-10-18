# Adversarial admission review: SAT assignment binding under renaming

16 October 2026. Model-generated, exact-input design review; not a human review, research rating or empirical result. No formulas were generated, checkpoint loaded, inference run, or future outputs inspected for this review. The two design inputs and source files were left unchanged. I coordinated design concerns with the draft's author; the judgment below is my own.

## Decision

**Conditional GO for the bounded truth-only construction gate and, if support exists, the narrowly described source-fidelity assay. No fatal conceptual contradiction was identified.** Settle the concrete interpretation and numerical details below before freezing implementation. This is not admission of a test that identifies a unique mechanism, separates semantic detectors from assignment-selective margins, or decides whether the published approximate interpretation is globally correct. Those stronger claims remain no-go regardless of the outcome of this bank.

The construction has a real prospective question: on fresh inputs selected without neural outcomes, do named threshold events move to the prescribed partner unit while the semantic repair is preserved? Success would add measured conditional scope to two published bindings. Failure would identify where the proposed practical extension fails. Neither result is automatically important enough to close the larger manuscript's significance gap.

## Exact inputs and review boundary

| Input | SHA-256 |
|---|---|
| `work/next_phase/sat_invariance_admission.md` | `4e5447713dffffb9c0379b632a406b183b403c25943667f4fdd35ec38fb896d2` |
| `work/next_phase/sat_renaming_design_draft.md` | `cbf4cbd011fb1664b11493c405087611c8dd683cf274f22ec81ec35a5fcbfac3` |
| `work/next_phase/sat_semantic_theory.md` | `e9c4f571ada0a9568b5fd0e455f2f2696869ce97739e20d058420dd3b1400566` |
| `work/next_phase/sat_published_neuron_rules.json` | `5768a10f40de3c3a5a8b981a571992bd4d7f201b5034c87817f3c99221e9510d` |
| `work/sat_case/upstream/model.py` | `7555e1f2f27546225397aa12de4fde1f6c903e9804766de477fe4c02770691ea` |
| `work/sat_case/runtime.py` | `bbc76d5f2d4a856944f54269a6da2d388f0ea6aee81ab03d9486e2e5c8bca230` |

Locations below refer to the frozen draft's line numbers. Earlier completed campaign reports were consulted only for the stopped comparison's scope. Primary-source reinspection covered Palumbo's approximation definitions, thresholded interpretation, assignment rules and known misses. This is an admission review, not a new novelty search or an independent runtime check.

## Strongest counterargument

The entire desired result can be produced by ordinary assignment-selective classifier margins. If a unit's scalar score tends to rise when its nominated assignment satisfies the clauses, an UNSAT-to-unique-SAT repair followed by variable renaming will select the corresponding partner. Nothing here distinguishes a semantic computational primitive from a learned numerical decision surface with that behavior. The interpretation and this alternative are compatible, not competing hypotheses.

The matched negative formulas sharpen only a restricted contrast. They receive the same token edit but remain UNSAT, whereas positive formulas become SAT. Consequently, context-dependent SAT confidence and semantic repairability remain confounded across cohorts. Equal oracle SAT labels between the two naming contexts also do not equalize native confidence. Four-unit selectivity can reject an assignment-blind global rise or fixed physical-unit response; a confidence-plus-syntax model can still implement the prescribed pattern.

Moreover, an approximate interpretation with no error guarantee on this selected distribution can accommodate either result. The proposed 0.90/0.80 thresholds are new operational expectations, not consequences of the source. The defensible contribution is therefore a prospective measurement of conditional validity and failure modes for two existing labels. It is not decisive evidence for the origin, uniqueness or explanatory superiority of those labels. The current draft largely respects this boundary; the eventual paper must preserve it rather than promote a successful benchmark into a mechanism-identification claim.

## What is and is not identifiable

**The narrow binding contrast is identifiable.** Unique satisfaction makes the partner assignment true and the wrong-pair assignments false at the repaired endpoint. The fixed correct, identity and cross-pair mappings use identical observations and weights. The exact positive pattern excludes vacuous all-zero commutation. Global endpoint deduplication and the quartet as one orbit prevent the most obvious artificial replication. These are substantive improvements, not defects requiring another cohort.

The controls distinguish specific behaviors. A shared scalar confidence passed through fixed monotone unit thresholds produces nested active-unit sets; it cannot produce the two exclusive, swapped singleton patterns. That restricted null can lose. Arbitrary confidence with syntax gates or assignment-specific margins cannot be ruled out. The negative cohort controls a context-insensitive response to the common edit, not every possible token/context mechanism and not a causal effect of readiness isolated from satisfiability.

The source defines distribution-dependent approximation, reports satisfying-assignment misses and derives a weighted clause-count account. It supplies neither universal renaming invariance nor a per-unit conditional error ceiling for this new cohort. Its aggregate hidden-state error is not an admissible substitute. These qualifications are supported by [Palumbo et al., §3, §4.2.3 and Appendix D.2](https://arxiv.org/html/2407.13594v2).

The admission memo's approximate commutation bound is a union-bound consistency identity: if the corresponding exact predicates agree, a bit mismatch requires at least one predicate error. Measuring both sides does not independently validate a mechanism. Pattern accuracy, transport agreement and mapping advantage also reuse the same four-unit events. Report them as complementary summaries, not independent replications or multiple independent confirmations.

## Adjustments before implementation freeze

### Major: make the decision rule answer the declared conditional question

**Lines 99–110.** Retain the proposed cutoffs as explicitly author-chosen finite-cohort diagnostics. Say “the prespecified source-extension criterion was not met,” rather than implying a violation of Palumbo's original approximation guarantee. A failed bootstrap advantage alone is insufficient evidence that identity or the wrong mapping is equally good; distinguish an observed reversal/tie from imprecision.

Freeze the integer equivalents: per pair, at least 58/64 primary positives, 404/448 primary negatives, 26/32 exact positive patterns, and at most 51/512 off-pair false positives. Freeze the quantile convention. Report the two mapping contrasts separately for each pair as well as pooled; require positive point advantages within each pair before describing **both** bindings as supported. The pooled interval is a pooled claim, not two separate interval claims.

Define “unrenamed selectivity is good but renamed selectivity fails” before outcomes. A simple bounded choice is to apply the same 0.90 sensitivity/specificity criteria separately in each direction, with their explicit denominators; otherwise call direction differences descriptive. This prevents a post hoc narrative of a special renaming defect. No new fit, sample expansion or formal hypothesis-testing hierarchy is needed.

### Major provenance fix: complete the prior-input ledger

**Line 43.** The named minimum is insufficient by itself. The parent's inventory identifies earlier native inputs in `work/sat_case/candidate_logits.npz:edited_clauses`, `work/sat_case/selected_decision_check.npz:edited_clauses`, and both `original_tokens` and `edited_tokens` in `work/sat_case/primary_witness.npz`. Include these alongside the pilot bank and mechanism queries. Their files are present; this review did not replay them. Convert stored clauses with the unchanged `pilot_common.py` encoder, verify existing token rows, deduplicate, and freeze all source-file and encoder hashes. Reject selected endpoints and their renamings against the complete resulting ledger before native extraction. The draft already requires a full inventory; this is the concrete completion needed to justify “fresh within the known campaign.”

### Major: make the selected population explicit

**Lines 39–47 and 76, 99–101.** A uniform proposal stream does not yield a uniform sample of native formulas after uniqueness screening, one-action selection, descriptor matching, first-compatible selection and global collision rejection. The target is an equal-weight mixture of the 32 selected action descriptors per pair, under this exact selection algorithm. State this next to the primary result, not only in limitations.

Replace “independent matched blocks” in the denominator table with “matched resampling blocks.” Removing shared endpoints resolves direct repeated-measure dependence, but the finite shared proposal pool and global exclusions do not establish probabilistic independence. The specified block bootstrap is a useful conditional descriptive interval; it is not exact randomization inference or a guarantee about the original formula distribution. Preserve eligibility and rejection counts, including failures of descriptor support. A support failure means this fixed construction budget did not produce the bank, not that the semantic family is impossible.

### Bounded numerical correction: replay the same batch context

**Lines 51–55.** Initial batches of 128 and a replay batch of 16 can create a numerical difference solely from batching. Use one complete matched block per batch: eight rows, with the same within-block order in both processes. This gives 64 initial batch calls and two replay calls while retaining exactly 512+16 input forwards. It also makes a failed replay easier to interpret. This is a protocol adjustment before freezing, not permission to rerun selectively after failure.

**Lines 57 and 103–110.** Explicitly propagate conservative numerical-interval ambiguity to an unresolved benchmark, including the mapping-advantage requirement. Freeze the bounding algorithm; intervals must not be silently ignored for a nominal pass. The radii remain heuristic sensitivity intervals, as the draft correctly states, rather than proven machine-error bounds.

## Feasibility and the additional output calculation

The retained literal has the stated algebraic effect: the edit changes the anchor's violated clause to satisfied while preserving satisfaction of that clause for its partner. Exact uniqueness and full UNSAT checks are necessary and already present. Excluding `{x1,x2}` prevents the adjacent swap from changing sorted literal positions. I found no logical contradiction among these requirements, but I did not generate witnesses or estimate support. The truth-only gate is therefore appropriate; 32 descriptors per pair within 100,000 draws remains an unmeasured feasibility condition. Do not loosen it after a shortfall.

The finite 528-forward budget, no model-based selection, no replacement after outcomes, all-four-unit readout, full-vocabulary task scoring and separate replay are reasonable. Task failures must remain in the unit analysis. The draft's exclusion of a new readiness forecast takes precedence over the admission memo's optional suggestion; adding that branch later would change this reviewed design.

**Lines 114–128:** static source inspection confirms the affine identity. `Unembed.forward` is multiplication by `W_U` without an output bias; the final block adds the MLP output to `resid_mid`; there is no final normalization. The stated `W_out` transpose is correct. Caching full final-position vectors solely for parity is acceptable with the explicit prohibition on searching other units.

Keep this output calculation secondary. Parity failure should invalidate the clamp interpretation while leaving independently sound primary bit measurements reportable, as specified. Preserve clamp logit gaps and label near-tie argmax changes numerically ambiguous. A float64 replay from saved float32 values is a defined affine intervention calculation, not a demonstrated patched-native float32 decision. Even a large clamp effect identifies a downstream contribution under this clamp, not natural mediation or a uniquely necessary assignment mechanism.

## How an outcome can change the explanatory decision

| Retained outcome | Permitted change in conclusion |
|---|---|
| Selective patterns and prescribed mapping advantage, with successful task repair | The two source labels are useful on this precisely defined intervention cohort; simple global-rise and fixed-unit accounts fail here. |
| Strong original-direction selectivity but failed renamed-direction criterion | A localized conditional transport limitation warrants investigation; this does not refute the source's original average-case account. |
| Both directions fail | The selected stress domain exceeds this small interpretation's measured useful scope; a special renaming defect is not isolated. |
| Task succeeds but named bits fail | These four threshold events do not faithfully summarize the model on these inputs; other representations may support the decision. |
| Named bits succeed but task fails | These selected events do not ensure the final task outcome. |
| Correct mapping commutes only because responses are zero or indiscriminate | No support for selective binding, even if raw transport agreement is high. |
| Support, replay or numerical adjudication fails | An unresolved bounded attempt, with preserved artifacts and no outcome-dependent redesign. |

No outcome of this experiment decides recursive-generalization origin versus arbitrary packing, uniquely distinguishes semantic detection from scalar margins, or rehabilitates the stopped unequal-supervision comparison. Those are contribution boundaries, not reasons to abandon an affordable, falsifiable conditional source check.
