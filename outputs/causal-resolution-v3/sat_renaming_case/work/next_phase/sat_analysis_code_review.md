# Focused SAT analysis-code review

16 October 2026. Model-generated static and deterministic bit-only review of contract §§5–9. No formulas, model inference, scientific responses, random-number calls or scientific bootstrap draws were used. The pipeline author owns all production-code changes; this reviewer changed no production files.

| Reviewed input | SHA-256 |
|---|---|
| `work/sat_renaming/analyze.py` | `a82356f38eafc86985607403e0354f765e14ff3ebffb16b27db097276a93471f` |
| `work/sat_renaming/protocol.py` | `98c4f843dd559c05fcd267a0434f1879f51ce79ecd1b996918b500c67469e20b` |
| `work/next_phase/sat_renaming_execution_contract.md` | `4b9dc2df3002dbbe56cbcc5d38c119ace801f16e2f3bd018d345deb4eda7af8c` |

**Verdict: no blocking arithmetic or adjudication defect found in this analysis version.** One minor reporting correction was sent to the author. This review does not substitute for the complete implementation freeze, independent formula-bank checker, native replay or actual support gate.

## Checks that support the verdict

- The `(block, cohort, state, unit, bound)` indexing agrees with the writer and expected target layout. Per pair the primary counts are 64 positive and 448 negative events; off-pair FPR uses 512 events. Direction counts are 32 positive and 224 negative. Exact patterns conjoin all eight nominated bits of their cohort.
- Mapping errors compare both endpoints of one origin; positive edited contrasts take cohort zero and the edited endpoint only. Identity and wrong-map errors share the exact bits used for the prescribed map. Signed changes retain negative values. All-zero outputs cannot pass selectivity or the strict mapping-advantage criteria.
- Off-pair false-positive bounds correctly reverse true-negative bounds. Per-pair raw-count benchmarks and directional criteria use the frozen integer thresholds. The robust logic requires every adverse bound to pass, identifies failure when a favourable bound fails, and otherwise reports numerical uncertainty.
- The bootstrap arithmetic samples indices at matched-block level within each fixed pair. Every metric, cohort and nominal/lower/upper value uses the same indices. Equal pooling occurs after the two pair means. Quantile indexing for the pooled lower endpoint is correct. No bootstrap was executed for this review.
- Full-vocabulary possible classes, ties, non-answer predictions and task bounds are preserved. Primary measurements do not filter task failures. The affine calculation uses the correct matrix orientation, corresponding original/renamed pre-edit values, both nominated/control pairs, and separate parity/near-tie treatment.

The independent script `sat_analysis_bit_review.py` AST-loads only pure functions and the relevant arithmetic statements; it never imports the formula protocol or invokes `main`. Its scalar reference checked every one of 6,561 admissible two-endpoint interval boxes, covering 65,536 total admissible endpoint assignments. A separate fully ambiguous signed-change box was checked against all 65,536 four-vector assignments. Full-vocabulary winner/tie/non-answer cases and four deterministic 64-block bit patterns checked indexing and the expected `PASS`, `CRITERION_NOT_MET` and `UNRESOLVED_NUMERICAL_SENSITIVITY` decisions. These are code fixtures, not synthetic scientific observations. The review runtime was NumPy 2.4.2; it does not replace the pinned scientific-runtime gate. The successful output is retained in `sat_analysis_bit_review_receipt.json`.

## Actionable minor correction

**`analyze.py:286`: denominator metadata mixes units.** For event metrics, `denominator` is an atomic count, such as 64 positive bits. For already normalized four-unit errors it is 32 matched-block averages even though a positive-cohort full-endpoint error includes 256 bit comparisons per pair, and an edited-only error includes 128. The computed rates are correct, but the common field can mislead readers and downstream tables.

Label each denominator's unit explicitly, or retain both the resampling-block count and primitive comparison count. Per cohort and block the full endpoint error averages eight bit comparisons, the edited endpoint error averages four, and the signed-change statistic averages four coordinate discrepancies. Balanced averages and differences also need an averaging-unit label rather than an implied binomial denominator. This is an output-description fix; it requires no scientific design or threshold change.

No external scholarly assessment is given. Admission remains limited to conditional transport of the two source-fixed bindings and the separately labelled affine accounting.

## Resolution and final reviewed versions

The author corrected the minor reporting issue before execution. I reinspected the replacement `denominator_metadata` routine: normalized mapping errors now identify matched-block aggregation and their primitive comparison counts; pattern, quartet, joint and balanced metrics carry separate explanatory metadata. The reviewed scientific arithmetic is unchanged. **The reporting issue is closed; no outstanding blocker remains within this focused review.**

Final inspected `analyze.py` SHA-256: `32eb8d671b735ead29e9aa2f6bc95917617ab09d3e265cf03e1c06639b663df5`.

Final inspected `protocol.py` SHA-256: `d33e9a59ca3c7f8df53a0a4c0ade1ff7e6949733aefc2eefd7003855e67b1464`. Its additional independent-bank-status and environment checks were read; the fixed statistical constants are unchanged.

The independent bit-only receipt above remains bound to the earlier arithmetic-equivalent analysis input and was not relabelled as a fresh execution on the revised file. The author's separate `work/sat_renaming/selfcheck.json` records 37 passing deterministic bit/logit/AST checks under NumPy 2.4.6 and explicitly binds both final hashes, with zero formulas, native forwards, fits and scientific bootstrap draws. I inspected that receipt without rerunning its checks.
