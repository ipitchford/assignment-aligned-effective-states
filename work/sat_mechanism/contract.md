# Fixed-score native transport in the published 2-SAT checkpoint

**Status: executable DEVELOPMENT contract, drafted 15 October 2026; not executed.** No new preactivation, edited-input or protected-test outcome was inspected while drafting this contract. It must be hashed before extraction begins. The old development assessment set has already been opened for other outcomes; this is not a protected confirmation. Execute this bounded study once, retain every failure, and stop at the decision below. Changes after extraction require a separately labelled exploratory version.

## Question and prior information

Does a negative preactivation at a source-nominated assignment detector encode whether that assignment violates exactly one clause, and does the **same fixed readiness score** predict activation of that same native unit after an appropriate legal clause replacement?

The published 34-bit report has a replayed same-report edit-response collision. That is not yet a policy impossibility or evidence that amplitude repairs decisions. In the frozen pilot, post-ReLU float magnitudes achieved decision success **0.622656**, versus **0.624512** for the 34 bits; the difference was **−0.001855**. Preserve this negative result. A scalar correlation with violation counts will be only a diagnostic; the central test is paired transport to actual native activation after input edits.

Reuse the checkpoint, original model computation, canonical ten-clause grammar, 4,096 formulas, 3,072/1,024 origin split, precision and numerical provenance in `../sat_case/contract.md`. Do not generate new formulas, open original training data, retrain the model, inspect other units, or access protected evaluation data. Do not silently change the original implementation or its published threshold `>=0.5`.

The fixed assignment identities come from Appendix G Table 2, transcribed in `sat_published_neuron_rules.json` and audited in `sat_native_mechanism_sources.md`. The 17 units are

`10,41,55,77,81,185,224,245,261,291,317,326,380,411,416,492,495`.

They cover 16 assignments: 55 and 416 both name TTTFF. Treat these as approximate published hypotheses, not proven identities. Do not substitute Table 3's multiple-assignment sets, rename a failed unit, or choose whichever assignment correlates best. The existing collision's newly satisfying assignment FTTTT is **outside** this singleton set; this study cannot be described as repairing that witness or the whole 34-unit explanation.

## Freeze before the first extraction

Archive hashes of this contract, the source mapping, checkpoint/source files, existing pilot bank and the complete implementation. Save the selected query list before extracting any preactivation. All origins stay in their existing split. Source hooks are `blocks.1.mlp.hook_pre` and `hook_post`, final readout position. Retain the original 34 published units and the nominated controls; do not save a searchable new whole-layer feature bank.

For every original formula, collect pre/post values in one unchanged float32 forward pass, CPU, eval mode, two threads. Check `post=max(pre,0)` and agreement with saved pilot postactivations; record exact maximum differences. Report all threshold distances. The 16 replay forwards are reserved for original origin indices 0–7 and, if Stage B runs, the first eight distinct edited token vectors in lexicographic order, in a separate process. If Stage B does not run, its eight replay forwards are unused. Near-threshold cases remain in headline denominators. Numerical replay is not exact-real verification.

## Stage A: one scalar readiness test

Let `s_i(F)` be unit i's original preactivation and `q_a(F)` the exact number of clauses violated by its published assignment a. The relevant stratum is `q_a>=1` and `s_i<0`: the assignment is false and its postactivation is exactly zero. The positive diagnostic label is `q_a=1`; the negative label is `q_a>=2`. Larger preactivation is the **frozen** proposed direction; never reverse it after seeing results.

Using original **fit** origins only, choose one threshold `tau_i` predicting readiness by `r_i=1[s_i>tau_i]`. Minimize balanced classification error on the clipped stratum. Candidates are all distinct observed float32 scores plus constant-zero/constant-one rules. Ties select fewer predicted positives, then the largest threshold. Store the actual comparison rule and threshold bytes. No regularization, feature selection, cross-unit mixing, second threshold or hyperparameter search is allowed. If either class is absent, use constant zero and record an unestimable fit. Fit the identical rule to postactivation as a same-capacity control; its score is constant on these rows.

On the unchanged assessment origins, report raw-score AUROC, frozen-threshold balanced accuracy, both class counts, clipped coverage and all unconditioned q=0/1/>=2 summaries for **every** source pair. Postactivation AUROC is exactly 0.5 on a nondegenerate clipped stratum. Average the two TTTFF unit results first, then macro-average the 16 assignments. Do not treat 17 units or repeated formulas as independent replications.

**Advance to native edits only if** every source pair has at least 20 assessment observations in each clipped q class, the assignment-weighted raw-score AUROC is at least 0.65, and at least 12 of 16 assignment scores exceed 0.60. These are practical development screening thresholds, fixed without preactivation outcomes; they are not significance levels. Report all results even if the screen fails. If it fails, stop this route—no extra origins, sign reversal, new units, alternative assignment sets or larger probe.

## Stage B: paired native transport of the frozen score

Select the following queries **before Stage A extraction**, using exact formula counts only. For each source pair and each existing split, take 32 origins with q=1 and 32 with q>=2. Order candidates by SHA-256 of the UTF-8 string `sat-native-v1|split|unit|qclass|origin_index`, where split is `fit` or `assessment` and qclass is `one` or `many`; take the first 32. If fewer exist, take all and record the shortfall. Never replenish after learning that an origin is unclipped or inconvenient.

For each selected origin choose one paired edit by the same minimum-hash rule, with suffix `|slot|kept_variable|new_variable`. Enumerate slots whose old clause is false under a. Keep either one of its false literals; choose a new variable outside the old clause's two variables. Form two canonical replacements sharing that same new variable pair:

- **Repair:** the kept false literal OR the new variable's literal true under a.
- **Neutral:** the kept false literal OR the new variable's literal false under a.

Both replacements differ from the old clause, obey the original grammar, use the same slot and old clause, and differ from one another in exactly one literal polarity. Repair changes q by −1; neutral leaves q unchanged. This matches syntax and edit size while varying the assignment-specific semantic effect. No score or edited outcome enters selection.

Run the two actual edited formulas through the unchanged native checkpoint. Deduplicate identical token sequences for computation, but retain all original query rows. The primary native outcome is `b'_i=1[post'_i>=0.5]`. Record `pre'_i>0` separately: crossing ReLU zero is not the published activation event. Retain logits and semantic SAT labels as descriptive context, not as selection filters.

For an action u, compute from its stated old/new clauses

`m_a(u)=1[a falsifies old]-1[a falsifies new]`.

The **primary structured forecast is fixed before any edited outcome**:

`bhat'_i = r_i(F) AND 1[m_a(u)=1]`.

Apply it to every selected query; report the clipped q>=1 stratum as primary and all selected rows separately. On the primary stratum it predicts readiness after repair and inactivity after neutral edits. Do not recalibrate `tau_i` using edited outcomes. The ideal symbolic readiness oracle `1[q_a=1] AND 1[m_a=1]` is a positive diagnostic of the approximate source interpretation, and is explicitly not a neural report.

### Fixed controls and measurements

1. Replace the original score with its post-ReLU counterpart, using the Stage A post-threshold. Same scalar capacity, same origins and actions.
2. Fix a cyclic derangement of the above ordered 17 source units: each unit receives the next unit's original readiness bit, with the last receiving unit 10. Use the target unit's true action effect `m_a`. This is the primary mismatched-unit control; no pair in this cyclic order shares the same source assignment.
3. Keep the correct readiness bit but replace `m_a` by the action effect for that next unit's assignment. This is the fixed mismatched-assignment control.
4. Fit one context-only logistic probe per target unit on the selected **fit** edited rows: slot/old/new one-hot lengths 10/40/40, intercept, L2 `C=1`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`, no tuning. Train and evaluate on the clipped q>=1 rows; if only one fit label exists, use that constant. This control gets action context but no formula, original preactivation, other-slot contexts or post-edit state. Retain convergence failures without trying new settings. Its greater parameter count is charged, not represented as capacity matched to the one-bit structured forecast.

Report each control's native response balanced accuracy, sensitivity, specificity and raw numerator/denominator. For each origin/unit pair retain the actual **repair-minus-neutral activation difference** `D_i=b'_i(repair)-b'_i(neutral)` (including −1 cases). Report mean D separately for q=1 and q>=2, and their difference. This semantic interaction is necessary: own-margin persistence or generic rectification information alone is insufficient.

Macro-average across 16 assignments after averaging duplicate TTTFF units. Provide 2,000 paired bootstrap draws of distinct original origin IDs, seed `2026101521`, carrying all that origin's units/actions together. Label intervals descriptive development uncertainty; do not claim independent neuron-level replication. Keep every selected origin in the archive, including unclipped, erroneous, unestimable and near-boundary cases.

For every metric, print the class/stratum support. A missing q class makes its AUROC/balanced accuracy NA. A missing native outcome class makes native balanced accuracy NA rather than applying a library's one-class convention. An assignment statistic is NA if either of its required unit statistics is NA; the 16-assignment macro is then NA and the corresponding admission gate fails for insufficient support. Never use `nanmean` or drop an unestimable unit to obtain a pass. Apply the same rule within each bootstrap draw; retain and count NA draws. Report a percentile interval from the valid draws only when at least 1,900 of the fixed 2,000 are valid; otherwise report the interval as NA. Do not replenish bootstrap draws. A missing interval does not change the point-estimate gate, but a missing point estimate always fails it. Other well-defined quantities, such as a supported mean repair-minus-neutral effect, remain reportable even when balanced accuracy is NA.

**Admit a promising mechanism-specific development result only if** each source pair retains at least 10 clipped assessment cases in each q class; the macro native repair-minus-neutral interaction between q classes is at least 0.25; the structured native balanced accuracy is at least 0.70; it exceeds each post-score and mismatched control by at least 0.10 and the context-only probe by at least 0.05; and the q-class interaction is positive on at least 12 of 16 assignments. Otherwise stop and report whether the failure was count decoding, native transport, specificity, context redundancy or coverage. These predetermined thresholds are triage criteria, not an earned causal theorem or protected confirmation.

## Costs, stopping and permissible conclusion

The fixed resource ceiling is 4,096 original forwards, at most 4,352 edited forwards (17 units × 128 origin/split rows × two edits), and 16 replay forwards: **8,464 before deduplication**, one small checkpoint, two CPU threads. No neural training, new feature bank, SAE, broader neuron search, decoder sweep, threshold sweep beyond the one stated fit rule, or formula replenishment. After Stage B, stop and return the complete receipts before any protected experiment or further search.

Each structured unit forecast has one calibrated comparison bit and the declared action descriptor. Preserving the whole published report plus all 17 readiness bits costs **51 fixed state bits**, even though the repeated TTTFF assignment is scientifically dependent. A single action costs 16 fixed bits (4 for slot and 6 each for old/new clause), or 14 packed bits for the 10×40×39 syntactic possibilities; the state-plus-action query therefore costs 67 fixed bits. This is only a cost ledger for the tested diagnostic; it is **not** an established repaired global SAT report. The 34 stored float32 pre/post values each cost 1,088 bits when used as reports. Charge model extraction, threshold constants, unit/assignment mapping, clause evaluation, action context and every fitted control's parameters and execution.

For comparison, full ordered clauses cost 60 fixed bits or 54 packed bits. A formula multiset costs **33 packed bits**, since C(49,10)=8,217,822,536, and suffices for all semantic edit queries given a valid old-clause context; unranking and SAT computation must be charged. It does not preserve sequence-sensitive native transformer outputs. Thus neither a 51-bit diagnostic nor a proposed 68-bit native refinement is globally compact for this semantic task. A complete 32-entry capped-count vector costs 64 fixed ternary bits or 51 packed bits, but is a symbolic oracle, not a learned neural feature. Counts capped at two answer one replacement; do not claim they form a closed state for arbitrary edit compositions.

The strongest objection is that a source-selected SAT detector may correlate with weighted clause composition and proximity to threshold without containing a useful assignment-specific reserve. The paired repair/neutral interaction, frozen semantic threshold, same-unit native outcome, mismatched controls and context-only comparison address that objection in one bounded study. They do not prove that negative values currently mediate the checkpoint's output: ReLU blocks that path. A positive result would refine a published detector account for a new prospective query, not refute it, prove universal distance-to-satisfaction semantics, repair all 34 units, or establish a broader generality. Broad claims that inactive preactivations contain predictive information also have prior art; novelty would need the specific native semantic transport and later consequential use.

**Deferred:** global SAT/edit-selection classifiers, a complete q decoder, new quantizers, and a stronger report-policy collision search. The 17 singleton rules cover only half the valuation space; a partial q decoder cannot certify global UNSAT. A same-report semantic response fork also need not force different optimal actions. A separate future policy audit could freeze the first 128 same-code/same-menu pairs with nonempty disjoint successful-action sets and check all 39 edits, but it is not executed or admitted by this contract and cannot substitute for the native-transport test.
