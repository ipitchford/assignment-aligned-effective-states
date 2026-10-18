# SAT assignment binding under renaming: narrow admission

16 October 2026. **Admit one bounded source-fidelity experiment:** does a controlled repair switch the responding native unit according to a fixed assignment renaming? This is a sharper question than persistence of one unit's own margin. It does not distinguish semantic detectors from every possible margin model, establish downstream mediation, or settle research significance. No inference, CNF generation, source mutation, or new fitting was performed for this memo.

The existing readiness study passed its development criteria. The continuous comparison used different supervision and sample counts, so its weaker result does not identify a unique mechanism. This proposed study changes the question to **semantic binding across units**, with no new learner or threshold calibration.

## What the published interpretation predicts

[Palumbo et al., ICML 2025](https://arxiv.org/html/2407.13594v2), §3, §4.2.2–4.2.3, Appendix D.2 and Table 2, supplies approximate Boolean predicates for thresholded neuron activation. For a singleton rule, write `b_i(F)=1[post_i(F)>=0.5] ≈ 1[q_a(F)=0]`. The source code's `>=0.5` boundary should be retained even though paper prose sometimes uses `>0.5`.

The source explicitly reports that every relevant high-mean assignment detector has some satisfying formula producing insufficient activation (D.2, “Interpreting Neurons via Decision Trees”). Its axioms bound error over an input distribution; the reported hidden-state error is about 0.182 for the general interpretation. They do not give a universal or conditional error guarantee for the new cohort. Under exact singleton semantics, corresponding bits must commute with renaming. Under approximate semantics, their mismatch probability is at most the sum of the two applicable predicate-error probabilities; those conditional probabilities must be measured here, not borrowed from the original aggregate.

**Not predicted:** equal raw preactivations, equal gains, exact violation counts, tied attention coefficients, or a permutation of the whole hidden layer. D.2 records asymmetric literal/head preferences and derives weighted rational clause-count contributions under canonicalized first-layer representations. Its finite trained coefficients need not respect a variable-renaming symmetry. Failure of raw-margin equality would therefore be uninformative. Failure of selected bit transport would test an extension of the source account, not refute its stated average-case result.

## Fixed mapping and one intervention family

Use only the involution `π: x1 ↔ x2`, leaving other variable names unchanged. The source-fixed pairs, verified by lookup in `sat_published_neuron_rules.json`, are:

| Unit i and assignment a | Unit j and assignment b = πa |
|---|---|
| 261 — TTFTT | 291 — TFTTT |
| 411 — TTFFT | 492 — TFTFT |

The four-state construction below checks both transport directions within each source orbit; renamed copies are not additional independent origins. These are unique singleton rules, avoiding the 55/416 duplicate. This is a partial transport: source units 77 and 245 rename to assignments without singleton rules, so a permutation of all 17 nominated units is not available. Do not map those units to more complex Boolean rules as though they were single-assignment detectors.

For each unordered pair, freeze the anchor orientation `(a,b)` before extraction. The positive cohort consists of **fresh initially UNSAT formulas repaired to the unique satisfying assignment a**, with the renamed copy repaired to b. Select entirely from syntax, exact truth tables, and violation counts:

1. Require a ten-clause formula F in the original distinct-variable grammar, with `M(F)=∅`, `q_a(F)=1`, and `q_b(F)>=2`. Its sole clause occurrence falsified by a is the edited occurrence c at slot k.
2. Require c to contain a kept literal ℓ with `ℓ(a)=0` and `ℓ(b)=1`. Choose a new variable outside c's variable pair, and replace c by `d = ℓ ∨ t`, where t is that new variable's literal true under a. Both c and d are true under b, so `q_b` stays unchanged, while `q_a` falls from one to zero.
3. **Explicitly require `M(F[k←d])={a}`.** Repairing a's unique violation alone does not prove uniqueness. Check it by enumerating the 32 assignments. Do not select on native predictions or activations.
4. To isolate renaming from literal reordering, exclude the unordered variable pair `{x1,x2}` from every clause of F and its edited endpoint. Swapping adjacent indices then preserves sorted literal positions. Preserve all ten clause slots. This restriction's available support is unmeasured; insufficient support must stop the plan rather than justify relaxing it after neural outcomes.

Match each positive origin to a fresh negative origin H with the **same action descriptor `(k,c,d)`**, `q_a(H)>=2`, and `q_b(H)>=2`. Apply the same grammar and kept-literal restrictions. Explicitly require `M(H)=M(H[k←d])=∅`; the target count staying positive alone does not prove the whole edited formula remains UNSAT. The kept literal makes `q_b` unchanged in both cohorts. This negative cohort distinguishes selective repair from responses to the common token edit, while the original and renamed positive copies hold the SAT transition fixed. Quotas and matching must be frozen before extraction, with no outcome-dependent relaxation.

Freeze a finite quota and deterministic hash tie-break before extraction. Count and retain semantic eligibility/exclusion totals. Keep an entire renaming orbit in one split and dependence cluster. Do not select a different permutation, assignment, unit or edit after observing failure.

The four concrete inputs per selected case are `F`, `F'=E(F)`, `πF`, and `πF'=π(E(F))`. Renaming means substituting variable names throughout formulas, assignments, and old/new clauses; it does not alter signs or slot indices. Exact algebra gives

\[
q_{\pi a}(\pi F)=q_a(F),\qquad
\nu_{\pi a}(\pi c)=\nu_a(c),\qquad
\pi(E_{k,c\to d}(F))=E_{k,\pi c\to\pi d}(\pi F).
\]

These identities follow from evaluating corresponding literals. The existing semantic theory supplies `q'_a=q_a−ν_a(c)+ν_a(d)`. They certify the intervention correspondence without assuming anything about numerical margins. They do not imply that the native network obeys the same correspondence.

## Native prediction, controls and falsification

The exact singleton ideal predicts for a positive case:

| Native bit | F | F′ | πF | πF′ |
|---|---:|---:|---:|---:|
| `b_i`, source assignment a | 0 | 1 | 0 | 0 |
| `b_j`, source assignment b | 0 | 0 | 0 | 1 |

For a matched negative case it predicts zero for all eight entries. In either cohort, the other pair's two source assignments are false at all four endpoints, so its units provide additional fixed negative controls without additional forwards.

Thus, with `Δ_i(F)=b_i(F′)−b_i(F)`, the expected response changes from `(Δ_i,Δ_j)=(1,0)` to `(0,1)` after renaming. Retain negative Δ values as failures. The empirical equivariance comparisons are `b_j(πG)≈b_i(G)` and `b_i(πG)≈b_j(G)` for both endpoints `G∈{F,F′}`. Equality alone is insufficient: an all-zero response commutes but fails the semantic prediction.

Compare the prescribed swap `(261↔291, 411↔492)` with an unchanged physical-unit mapping and one frozen cross-pair wrong mapping, `(261↔492, 291↔411)`. Score all three mappings on the **same four-unit vector** at the same endpoints, with identical case weights and error definitions; do not fit the mappings or compare two-unit and four-unit errors. Unique satisfaction ensures that the wrong target assignment is false, so this tests identity rather than merely whether some unit increases after a successful edit. The SAT/UNSAT transition is identical in both naming contexts; a global confidence increase cannot explain selective switching by itself.

For `N+` positive and `N−` negative source-origin orbits **per unordered pair**, put `N=N++N−` and report. These counts include each orbit once, although its four observed states test both directions:

- All eight primary native bits per origin: expected-positive events (denominator `2N+`) and expected-negative events (`6N+ + 8N−`). Report sensitivity and specificity separately, plus exact eight-bit pattern agreement (`N`), also separated by cohort; ordinary accuracy alone can reward zeros.
- Four endpoint equivariance comparisons (`4N`), and the two response-change comparisons (`2N`). Report both alongside nonzero selectivity so vacuous commutation cannot pass.
- For the full four-unit control vector, `16N` semantic bit events, with `2N+` expected positives and `14N+ + 16N−` expected negatives. Correct versus unchanged/wrong mapping transport errors each use `8N` endpoint comparisons (four units times two endpoints), or `4N` response-change comparisons. Report paired differences and positive-cohort differences, since all-zero negative responses commute under every mapping.
- The four native task classifications (`4N`): positive oracle labels UNSAT, SAT, UNSAT, SAT; negative labels all UNSAT. Report complete four-input correctness (`N`) and separate cohorts. Do not filter the unit analysis to correctly classified cases.
- Cluster dependence by original formula/renaming orbit; four units or repeated endpoints are not independent replications. Retain every threshold-boundary case, missing cell and exclusion. Record matched-origin identifiers; if several origins share a source orbit, keep that dependency explicit rather than treating the matches as independent.

The working extension fails when responses do not switch to the source-named partner, when both units rise indiscriminately, or when the supposedly repaired assignment fails to activate its nominated detector. A negative result is strongest when ordinary unrenamed selectivity works but the renamed counterpart or paired mapping advantage fails. If both versions fail, the selected stress domain may expose the source approximation or parser, rather than a special binding defect. If only task classification succeeds, the network may use other neurons. Report these separately; do not rescue the result by relabeling units or changing the cohort.

No universal numerical acceptance threshold follows from the source. A future contract should fix practical error ceilings and support minima before extraction; inference must not start from this memo alone. In the positive cohort alone `q_a=1`, so readiness is constant by oracle construction. The matched negative cohort permits a secondary discrimination test using the **already-frozen readiness thresholds**, transferring unit i's forecast on the original origin to partner j's edited renamed response. Such a test must use no recipient prestate, new fitting, threshold adjustment, or changed supervision comparison. The old thresholds were developed on a different, initially SAT domain, so transport here has no guaranteed error bound. This does not establish a report-only edit-selection policy or a unique role for negative-score precision; switched native selectivity remains the primary question.

## Nearest prior work and strongest objection

Palumbo's assignment-detector account is the closest prior and already approximately predicts this bit-level behavior. [Selsam et al., *Learning a SAT Solver from Single-Bit Supervision*, ICLR 2019, §3](https://arxiv.org/pdf/1802.03685v4) explicitly builds variable/clauses/literal permutation and negation invariance into NeuroSAT. [Es-sebbani et al., 2026, §3.5 and Appendix F](https://arxiv.org/html/2602.12665v1) uses renaming and symmetric duplication as logical-reasoning stress tests. [Geiger et al., JMLR 2025](https://jmlr.org/papers/v26/23-0058.html) supplies the established intervention-correspondence framework. None is evidence that this checkpoint's proposed experiment has already succeeded; equally, failure to locate it is not evidence of originality.

**Strongest counterargument:** a collection of ordinary assignment-selective classifier margins can produce the entire expected pattern. That alternative is compatible with semantic binding. A positive result would exclude simple physical-unit persistence or global SAT-confidence accounts on this cohort, while strengthening the published assignment interpretation under controlled edits. It would not identify a unique computational implementation, show negative values mediating the current output through ReLU, or establish the global abstraction/decision repair sought by the larger paper. The admission is for that bounded scientific distinction, not an automatic contribution or increase in broader significance.

Inputs read: `sat_mechanism/REPORT.md`, `sat_margin_baseline/REPORT.md`, the native-rule and negative-preactivation source audits, published rule JSON, and `sat_semantic_theory.md`. Primary reinspection covered Palumbo §3, §4.2.2–4.2.3, Appendix D.2/Table 2; NeuroSAT §3; Es-sebbani §3.5/Appendix F; and the JMLR primary record. Six targeted searches on 16 October 2026 used: `"Palumbo" "2-SAT" "permutation"`; `"mechanistic interpretability" "variable renaming"`; `"NeuroSAT" "permutation" "negation"`; `"mechanistic interpretability" "variable renaming" neuron`; `"2-SAT" "equivariance" transformer neuron`; `"mechanistic interpretability" "symmetry" "interchange"`. Only primary sources support this memo. Search was bounded; no global novelty claim is made.
