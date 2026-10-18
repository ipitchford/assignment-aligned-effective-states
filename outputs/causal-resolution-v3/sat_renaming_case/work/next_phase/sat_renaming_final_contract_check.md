# Final-contract admission check

16 October 2026. Model-generated static review, bound to `work/next_phase/sat_renaming_execution_contract.md`, SHA-256 `4b9dc2df3002dbbe56cbcc5d38c119ace801f16e2f3bd018d345deb4eda7af8c`.

**GO for implementation and the prescribed staged freeze. All required pre-execution fixes from my admission review are captured.** This does not authorize bypassing the frozen implementation, truth-only support, independent-oracle, replay or numerical gates. Feasibility remains unmeasured. No formulas, model inference, bootstrap draws or reruns were performed for this check; no reviewed files were modified.

- The complete known prior-input ledger now includes the three previously omitted sources, with conversion, independent decoding and hash requirements (§3).
- The 32-descriptor quota, 100,000-proposal cap, fixed matching and no-rescue stop remain unconditional. Conditional equal-descriptor scope and descriptive block resampling replace the earlier independence implication (§§3,5–6,10).
- Initial and replay calls use the same eight-row batch contexts. Both per-pair mapping point advantages are required, alongside the two pooled interval conditions (§§4,6).
- The benchmark is explicitly author-chosen and finite-cohort. Failure of it is not presented as violation of the original source guarantee. Directional good-selectivity thresholds are correct: each direction has 32 true-positive opportunities and 224 true-negative opportunities per pair; the 0.90 ceilings give at least 29 and 202, respectively (§6).
- The clamp remains secondary, with parity, gap and possible-class reporting; failure does not invalidate independently sound primary measurements (§§7,9).

Section 9 supplies a conservative adjudication algorithm. The threshold-set boundary respects `>=0.5`. Enumerating the paired four-bit vectors preserves shared bits in each mapping contrast. Averaging per-case extrema, then taking monotone linear quantiles with the fixed bootstrap indices, bounds every admissible global assignment even when different extrema cannot occur together. The adverse-bound pass / favourable-bound failure / otherwise-unresolved rule is valid and intentionally conservative. Full-vocabulary possible-class sets conservatively retain ties. None of this converts the chosen radii into proven machine-error bounds.

One implementation clarification has been sent to the pipeline author: the off-pair statistic is a **false-positive** count, so its bounds invert the correctness bounds. With 512 events, `FP_lower = 512 − TN_upper` and `FP_upper = 512 − TN_lower`. Robust pass requires `FP_upper <= 51`; robust failure of that condition requires `FP_lower > 51`. Summing correctness indicators directly would instead give specificity. This clarifies the existing fixed benchmark and does not require changing the contract or adding an experiment.

The bounded claim remains conditional transport of two existing assignment labels. No result identifies a unique semantic mechanism, separates the labels from assignment-selective margins, or rescues the stopped comparison. Implementation and analysis code still require fidelity checks against this contract before execution. I remain available for that code review.
