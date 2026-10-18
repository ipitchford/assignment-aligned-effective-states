#!/usr/bin/env python3
"""Apply the evidence-led v4.4 revision after the exact-hash v4.3 review."""
from pathlib import Path


PATH = Path("outputs/causal-resolution-v3/paper/manuscript.md")


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


text = PATH.read_text()

front_end = text.index("---", 4)
front = '''---
title: "A Sparse Native Assignment Report Spans the Effective Final-MLP State of a 2-SAT Transformer"
subtitle: "Protected label tests, geometry-matched basis controls and exact readout accounting"
date: "18 October 2026 · Version 4.4 working draft · Unrefereed"
lang: en-GB
abstract: |
  Cammarata's *Neural Networks Are Already Clean* proposes that native neurons can be structured building blocks even when broader representations look distributed. We audit one favourable but capacity-unconstrained case: a width-512 final MLP in a five-variable 2-SAT transformer, for which Palumbo et al. selected 34 units by direct SAT-output effect and published singleton-assignment labels for seventeen. On a protected 7,252-formula bank, the singleton labels rank strongly (macro AUROC 0.991; unit range 0.961--1.000), but the frozen combined and threshold-fidelity criteria are not met: two units exceed a specificity limit by one of 32 cases, and four fail a positive cell. A protected full-layer extraction finds 70 numerically nonzero columns but rank 34. Retrospective accounting shows that the source 34 contain effectively all post-ReLU energy; the other 36 active columns contribute only $7.0\times10^{-53}$ of squared norm. Within that 34-dimensional live state, native one-to-one assignment matching remains far above 63 dimension-matched rotations (mean AUROC 0.984 versus maximum 0.915) and positive mixtures (maximum 0.954), while prevalence-preserving shuffled labels and one-violation controls reverse the comparison. Orbit-grouped linear probes reach mean AUROC 0.996 and recover three assignments with weak single native axes, separating native single-unit legibility from information present in the state. The final readout is exactly additive in unit activations: all 34 live units have positive SAT-minus-UNSAT coefficients, and zeroing the singleton seventeen or all 34 lowers accuracy from 99.49% to 61.33% or 20.06%. Joint argmax changes exceed summed individual changes because redundant positive evidence crosses decision margins, not because the affine readout contains interactions. Source-threshold misses and false positives still contribute useful logit margin, so the Boolean report discards decision-relevant magnitude. The same threshold report guides clause-edit choice but cannot determine every legal edit response. This is a topical single-checkpoint audit, not evidence about prevalence, developmental origin, capacity-driven superposition or an upstream causal route.
---'''
text = front + text[front_end + 3:]

text = text.replace(
    "| A sparse source-nominated bank spans the assignment report and supports the decision | Complete 512-unit extraction; protected matching against 63 fixed Haar rotations; post hoc bank restrictions; exhaustive individual and joint zeroing at the final affine readout | All 32 protected assignment matches lie in the source-nominated 34-unit bank and agree with its rule inventory; excluding that bank collapses matching to chance. All and only those 34 units affect task argmax on this bank, and joint zeroing lowers accuracy to 20.06%. The result is conditional on one checkpoint and final-cut interventions |",
    "| The effective final-MLP state is a sparse native assignment report | Complete 512-unit extraction; protected matching against 63 fixed full-space rotations; retrospective rank, energy, live-subspace, positive-mixture, prevalence and linear-probe controls | The source-nominated 34 units contain effectively all post-ReLU energy and span the rank-34 state. Native assignment matching remains above every 34-dimensional rotation and positive-mixture reference, while state-level linear probes recover the three weak single-axis assignments. This is one checkpoint and bank, with source selection fully disclosed |",
)
text = text.replace(
    "| Published assignment units contribute directly at the final linear readout | Exact zero ablation and within-stratum resampling for all seventeen units; all-unit zero-ablation census; earlier four-unit origin-restoration pilot | Individual published-unit zeroing changes 17--153 argmaxes and joint zeroing lowers accuracy to 61.33%; all 34 source-report units have an effect and the other 478 do not. Earlier restoration changes 115/126 initially correct edited decisions. These final-cut interventions do not identify an upstream route |",
    "| The live report contributes at the exact final affine readout | Exact individual and grouped zeroing; per-threshold-category margin accounting; within-stratum resampling; earlier four-unit origin-restoration pilot | Every live report unit has a positive SAT-minus-UNSAT coefficient. Joint removal crosses more argmax margins than individual removals because the logit contributions are additive and redundant. Earlier restoration changes 115/126 initially correct edited decisions. These final-cut interventions do not identify an upstream route |",
)

old_rank = "Among the 17 saved units, each nominated unit ranks first for its label except unit 55, which ranks second to unit 416 carrying the same published TTTFF label."
new_rank = "Against all 512 units, thirteen nominated units rank first for their label and four rank second: unit 81 is behind 334 for FTTTF, 261 behind 374 for TTFTT, 411 behind 201 for TTFFT, and 416 behind 55 for the shared TTTFF label. Thus units 374 and 201, used as activity-profile comparisons in the earlier pilot, are stronger same-assignment axes rather than semantic nulls. The 32-by-34 matrix in Figure 3 reports the complete source-bank competition."
if old_rank not in text:
    raise RuntimeError("rank sentence not found")
text = text.replace(old_rank, new_rank)

central = r'''## The effective final-MLP state is a sparse native assignment report

Palumbo et al.'s report $J$ contains 34 final-MLP units selected because each unit's largest direct effect on the SAT output exceeded 0.1 on the source training data. The source then fitted shallow logical rules to those units. The seventeen-unit cohort used above has a designated singleton assignment for each unit, while the source's broader decision-tree inventories can associate some of those same units with one to three Hamming-adjacent assignments. The other seventeen units have conjunction, exclusion or multi-assignment rules. The label audit fixes the designated singleton labels; the full-layer study asks how the complete state is organised.

### Protected full-layer comparison

The protected extraction covers the same 7,252 inputs, with the earlier seventeen columns and task logits treated as anchors. Every anchor is bitwise identical. A separate eight-row fresh-load replay reproduces preactivations, post-ReLU states, residual streams and logits bitwise on those rows. On the complete bank, 442/512 units are exactly zero, 70 are numerically nonzero, and the post-ReLU matrix has rank 34.

Before full-layer model access, the study fixed 63 Haar-distributed 512-dimensional rotations and two primary one-to-one matching statistics. The first matches all 32 assignments; the second matches the sixteen assignments absent from the published singleton-label set. A descriptive secondary restricts positives for each of those sixteen assignments to SAT formulas with no covered solution. The raw finite-reference ranks are 1/64 for both primaries; applying the frozen two-test Holm rule gives 1/32. These are extremeness ranks inside a fixed reference ensemble, not randomisation $p$ values, because sparse nonnegative native coordinates and dense signed rotations are not exchangeable.

| Protected matching target | Native mean AUROC | Full-space Haar mean | Best of 63 | Native minus best | Reference rank |
|:--|--:|--:|--:|--:|--:|
| All 32 assignments | 0.984233 | 0.953018 | 0.957967 | +0.026266 | 1/64 |
| Sixteen labels outside the singleton set | 0.974041 | 0.949115 | 0.956669 | +0.017372 | 1/64 |
| Exclusive-positive secondary | 0.978782 | 0.973042 | 0.977960 | +0.000823 | 1/64, descriptive |

The count descriptives in this protected comparison are partly mechanical. Native coordinates supply only 34 effective axes, whereas a 512-dimensional rotation produces hundreds of dense mixtures of the same rank-34 signal. Native therefore has fewer axes above 0.90 and 0.95 than every full-space rotation, while having many more above 0.98 and 0.99. The weakest native assignment axis also loses to every full-space rotation. The three weakest assignments---FTFFF, TFTFF and TTTFT---are exactly the three omitted by the source's two-leaf dedicated-rule cover. The exclusive-positive margin is only 0.000823, about 0.3 of the rotation standard deviation, so the 1/64 rank must not be read as a large effect.

The protocol's authoritative conclusion is reproduced verbatim:

> Native coordinates exceed the fixed Haar references on both primary matching statistics; the 32-assignment primary includes 16 assignments selected for published native units. The native-over-Haar matching advantage is also observed in the descriptive exclusive-positive secondary (uncovered assignments, SAT formulas with no covered solution; matched axes may include published units; not multiplicity-controlled). This is a within-checkpoint, within-bank single-axis-separability comparison against Haar references whose exchangeability null is generically false for nonnegative sparse ReLU coordinates; it does not separate assignment-specific alignment from generic coordinate privilege (see the clause-presence and SAT/solution-count control gaps) and is not evidence that any coordinate is causally used.

![Protected full-layer native-basis and exact-readout results. **A:** labelled dark points are native one-to-one assignment-matching means; grey segments show the full range over 63 fixed full-space Haar rotations and blue ticks their means. The exclusive-positive result is descriptive. **B:** four differently structured controls reverse direction. **C:** teal bars show the task-accuracy drop from individually zeroing each published unit at the exact final affine cut; the orange line shows mean argmax changes under within-stratum activation resampling. Joint zeroing of all seventeen units lowers accuracy from 99.49% to 61.33%. The full-space basis comparison is the protected historical test; the geometry-matched retrospective controls below determine its interpretation.](figures/sat-full-layer-basis-v3.pdf){width=100%}

### Retrospective effective-state and geometry controls

Opening the full result reveals that $J$ is not a small sub-bank inside a broadly active layer. Its 34 columns contain $1-2\times10^{-16}$ of the total post-ReLU squared norm; the other 36 numerically nonzero columns contribute only $6.98\times10^{-53}$. Their largest activation is $1.03\times10^{-24}$. The singular spectrum has 34 values from 51.45 down to 4.08, followed by values at or below $5.97\times10^{-15}$; the rank remains 34 for relative tolerances from $10^{-12}$ through $10^{-4}$. Thus $J$ is the effective live state of this final MLP on the bank. Post hoc exclusion of $J$ collapses assignment matching because it removes essentially the entire state, not because alignment was localised against a substantial non-$J$ alternative.

This structural correction does not erase the native-axis result. Repeating the one-to-one analysis *inside the live state* gives the following retrospective descriptive comparisons:

| Reference family, 63 draws | Native all-32 mean | Reference mean | Best reference | Native minus best | Weakest native minus best reference |
|:--|--:|--:|--:|--:|--:|
| Haar rotations within $\operatorname{span}(J)$, 34 dimensions | 0.984233 | 0.901504 | 0.915146 | +0.069087 | +0.021557 |
| Haar rotations of the 70 numerically active columns | 0.984233 | 0.924260 | 0.932969 | +0.051265 | $-0.011003$ |
| Dense nonnegative mixtures of $J$ | 0.984233 | 0.904640 | 0.915189 | +0.069045 | +0.021916 |
| Two-sparse nonnegative mixtures of $J$ | 0.984233 | 0.928814 | 0.954067 | +0.030166 | +0.098758 |

The native mean exceeds every reference in all four families. Within the exact 34-dimensional live span, even the weakest native matched axis exceeds every rotation and dense positive mixture. The same comparison on the 2,543 unique base formulas yields native 0.985665 versus a best 34-dimensional rotation of 0.919789. These families remain constructed references, not exchangeability nulls or learned alternatives.

Two controls constrain the interpretation. Independently permuting each assignment-label column preserves its exact 3.5--7.3% prevalence but destroys its semantics; native mean matching is then 0.5159 and loses to every in-span rotation (range 0.5321--0.5369). For the 32 labels $q_a=1$, meaning that assignment $a$ violates exactly one clause, native mean matching is 0.5530 and again loses to every rotation (0.6032--0.6151). Sparse native coordinates therefore do not generically win for equally rare labels or nearby clause-count events.

An orbit-grouped five-fold linear probe on the 34-dimensional state reaches mean AUROC 0.9956 and minimum 0.9842. For the three assignments with weak native single axes, probe AUROCs are 0.9898 (FTFFF), 0.9924 (TFTFF) and 0.9859 (TTTFT), compared with best-unit AUROCs 0.9029, 0.8809 and 0.8941. The complete state contains those assignments linearly even though no dedicated native axis separates them as cleanly. This distinguishes a clean native report from a claim that every assignment has its own Boolean neuron.

All 32 axes selected by the protected match belong to $J$, and each matched assignment appears in that unit's source tree inventory. Two matches in the supposedly uncovered half use published singleton units 416 and 411; the protected second primary excluded their designated labels, not their axes or the broader public source rules. Excluding the singleton seventeen leaves fourteen of sixteen uncovered matches unchanged; only TFTFF and TTTFT collapse, from 0.881 to 0.542 and from 0.894 to 0.641. These source-linkage facts are retrospective consistency checks, not held-out discovery.

![Semantic, geometric and readout views of the effective final-MLP state. **A:** positive-direction AUROC for the 34 source-report units against all 32 satisfying assignments. Orange boxes mark assignments in the source tree inventories; white circles mark protected one-to-one matches. **B:** retrospective uncovered-assignment matching on the complete linked bank and unique base formulas. Removing $J$ removes effectively all post-ReLU energy, so the resulting chance-level score is structural. **C:** exhaustive individual zeroing on the 7,252-input bank. The same 34 live units are the only units that change a task argmax. Joint zeroing values are exact at the final affine cut.](figures/sat-sparse-report-v5.pdf){width=100%}

### Exact readout accounting: additive logits and redundant margins

For the last MLP state $h$, the logits are exactly $z=rW_U+b_{out}W_U+hC$. There is no interaction among units at this cut: zeroing a set subtracts the sum of its unit-level logit contributions. All 34 effective units have positive SAT-minus-UNSAT coefficients, ranging from 5.813 to 11.963. The final MLP therefore supplies positive SAT evidence; the residual-plus-bias path predicts UNSAT on almost every bank row when all 34 contributions are removed.

Individually zeroing the published seventeen changes 17--153 argmaxes per unit. Jointly zeroing them changes 2,771/7,252 decisions and lowers accuracy from 99.49% to 61.33%. Their individual change counts sum to 2,131 and their union covers 2,071 rows. The joint count is larger because several additive positive contributions can jointly cross an argmax margin when no single removal does. This is redundancy after thresholding logits to a decision, not interaction in the readout. The pair 81/334 illustrates it cleanly: their individual counts are 17 and 69, their union is 78, and joint zeroing changes 140 decisions. The earlier 55/416 example is subadditive relative to the individual-count sum and is no longer used to support this point.

Jointly zeroing all 34 changes 5,764 decisions and reduces accuracy to 20.06%. Among baseline-correct cases it flips 2,720/2,720 covered-only cases, 2,050/2,050 cases with covered and uncovered solutions, and 992/994 uncovered-only cases from correct to incorrect; the two reverse changes are UNSAT inputs. Because $J$ is the whole effective final-MLP state, this is a layer-level ablation, not evidence that a small sub-bank uniquely carries the task.

The graded claim rests on contribution magnitudes rather than joint-count arithmetic. Across the seventeen published units, mean SAT-minus-UNSAT contributions on source-threshold false negatives range from 1.72 to 4.76 logits. Where false positives exist, their means range from 5.14 to 8.75 logits. Some of these rule-inconsistent and sub-threshold activations support correct SAT decisions; zeroing them can turn a correct prediction into an error. The thresholded logical report is therefore a useful coarse description, while the affine readout uses continuous positive evidence.

Within-assignment SAT/solution-count resampling changes only 0.0004%--0.3452% of argmaxes on average across the published units. This preserves task labels and much of the assignment signal, so it is a sanity check rather than a mechanistic control. At this final affine cut the total intervention is exactly the direct-logit effect and no downstream self-repair is possible; zeroing can still be off the natural manifold. None of these results identifies how the activations arose or whether the same route is used upstream.
'''
text = replace_between(text, "## A sparse source-nominated assignment bank supports the final readout", "## The edit target is not a function of current truth or the bit report", central)

text = text.replace("## The first action-selection pilot did not favour post-ReLU magnitudes", "## The first action-selection pilot showed no robust ordering between bits and magnitudes")
text = text.replace(
    "activity levels one, two and three account for 444, 227 and 56; levels five through ten account for none.",
    "activity levels one, two, three and four account for 444, 227, 56 and one; levels five through ten account for none.",
)

synthesis = r'''## What the SAT studies establish on their declared cohorts

The SAT studies identify the checkpoint's effective final-MLP state as a sparse native assignment report with target-dependent limits. Across all seventeen designated singleton units, every protected threshold-free ranking gate passes, while the frozen combined and threshold-fidelity decisions do not: two units exceed a specificity limit by one case and four fail a positive cell. Errors concentrate near the semantic boundary.

The protected full-space comparison placed the native assignment matches above 63 fixed Haar references. Retrospective analysis changes the explanation and strengthens the like-for-like result. The source-nominated 34 units contain effectively all post-ReLU energy and span the rank-34 state. Within that live span, native assignment matching still exceeds every dimension-matched rotation and positive-mixture reference. Prevalence-preserving shuffled labels and one-violation controls reverse direction, while orbit-grouped probes show that all assignments are linearly present even when three lack equally clean single native axes. This supports unusually legible native axes *within this effective state*, while leaving prevalence, development and capacity dependence unresolved.

The final readout is affine. All 34 live units add positive SAT evidence, and grouped removals cross more argmax margins than individual removals because that evidence is redundant. Threshold false positives and misses can still contribute useful margin, demonstrating that Boolean labels lose readout-relevant magnitude. The exact intervention establishes final-cut dependence, not an upstream route. Separately, the 34 threshold bits guide edit choice but cannot determine every legal clause replacement.

| Proposed use | Available information | Result and limit |
|:--|:--|:--|
| Predict a legal edit from the current published report | 34 bits and the common action | Current truth does not determine the target. Adding it removes 1,000/399,360 empirical errors attributable to report information; 1,484 logical target errors remain. The report-specific rate is occupancy-dependent. |
| Select an UNSAT-producing edit | Initial report or post-ReLU magnitudes and one action menu | Published bits guide choices above action context alone. A one-seed post hoc factorial changes sign with action encoding, so no general bits-versus-magnitudes ordering follows. |
| Forecast a named unit after a paired edit | Fitted readiness bit and assignment-specific action effect | 90.95% balanced accuracy; the readiness bit adds 12.24 points over semantic action effect alone. This is an opened-bank predictive result. |
| Test all designated singleton-assignment labels | Seventeen fixed units, five semantic strata and the source threshold | All ranking gates pass. A specificity guard fails for two units and four fail a positive cell. All 36/544 threshold false positives occur on Hamming-one SAT negatives. |
| Characterise the complete final-MLP state | All 512 coordinates; protected full-space rotations; retrospective energy, in-span, positive-mixture, prevalence and probe controls | The source 34 span effectively the whole rank-34 state. Native matching exceeds every like-dimensional and nonnegative-mixture reference; rare shuffled and one-violation labels reverse direction. Three assignments are much better recovered by state-level probes than by one native axis. |
| Test selected labels on consequential matched edits | Four-direction protected pilot with fresh quartets | Three directions detect at least 31/32 positives at 0.5; unit 411 detects 24/32. Units 374 and 201 are same-assignment axes, so the activity-profile comparisons are not semantic nulls. |
| Account for the exact final readout | Complete state, exact individual and grouped zeroing, category-level margin accounting | The readout is additive. All 34 live units have positive SAT-minus-UNSAT coefficients; grouped removal lowers accuracy to 20.06%. Rule-inconsistent and sub-threshold activations can support correct decisions. |

The approximation in the source already allows distribution-dependent errors and weighted assignment margins. The findings delimit its reuse rather than contradict an exact guarantee. The original FTTTT edit witness remains logically unresolved even though unit 189 is a strong FTTTT axis: no current-state unit can restore information about which clause replacement will be applied. The stopped negative-score screen remains in the record because its gates were structurally foreclosed, not as evidence against inactive features.
'''
text = replace_between(text, "## What the SAT studies establish on their declared cohorts", "# Supporting exact calibration", synthesis)

alignment = r'''# Implications for alignment research

The decision-relevant lesson is methodological. A native representation can be far more axis-legible than alternative bases inside the same effective state, while still losing information when converted to Boolean feature flags. In this checkpoint, the source 34 are not a sparse island in a 512-dimensional live layer; they are the entire effective final-MLP state on the bank. Within that state, however, native axes outperform dimension-matched rotations and positive mixtures on assignment labels, and the advantage disappears for prevalence-matched shuffled labels and one-violation controls. This is a sharper result than the original full-space rotation comparison and a more limited one than a claim about frontier models.

The exact readout gives a second lesson. The layer is linear at the intervention cut, yet joint removals produce nonlinear changes in discrete decisions because redundant positive evidence crosses argmax margins. A feature monitor that stores only threshold bits would hide contributions from sub-threshold and rule-inconsistent activations. For safety work, continuous feature values, readout coefficients and decision margins may matter even when the underlying axes are semantically clean.

The report also separates semantic legibility from action sufficiency. Its bits describe the present SAT computation and guide clause edits, but they do not determine every edited response. Any proposal to use internal features for monitoring or control therefore needs a contract: target, intervention cut, retained state, side information, decoder class, tolerance and evaluation distribution. Claims about deception detection or behavioural certification require direct tests of those targets; this toy does not supply them.

Four gates follow for native-feature alignment studies:

1. **Semantic stress test.** Measure sensitivity and specificity on hard negatives, symmetries and compositional cases, preserving continuous scores alongside operational thresholds.
2. **Effective-state accounting.** Report live energy, rank and the feature-selection channel before interpreting localisation. Compare native axes with alternatives inside the same live subspace and with geometry- and prevalence-matched controls.
3. **Exact causal accounting.** Separate additive logit effects from nonlinear decision changes. Measure individual, grouped and complement interventions, and distinguish final-cut totals from upstream natural routes.
4. **Action-conditioned sufficiency.** Test the report on the actual monitoring or control decision. Use collision witnesses and restricted-decoder certificates to distinguish missing information from an inadequate reader.

The decisive next experiment is a frozen seed-and-width sweep of the same SAT family, with independent unit nomination inside each checkpoint. Widths below, near and above the 32 assignment atoms would reveal whether dedicated native axes persist under capacity pressure. Each run should report live-unit count, effective rank, native-to-probe gaps, in-span reference comparisons, threshold loss and grouped necessity on both source-distribution and engineered hard-negative banks. That experiment directly tests the capacity account that this width-512 checkpoint cannot.

Upstream interchange or path-specific interventions are also needed to determine whether the clean final report is assembled through a stable route. Native units, sparse-autoencoder features and compositional groups should then be compared under the same information, transformation and decoder-cost ledger. The useful question is which representation supports a frozen safety-relevant prediction or intervention, not which looks cleanest in isolation.
'''
text = replace_between(text, "# Implications for alignment research", "# Conclusion", alignment)

conclusion = r'''# Conclusion

Cammarata's proposal becomes testable when “clean” is divided into label ranking, threshold fidelity, basis legibility, causal use and action sufficiency. In this checkpoint, published singleton labels rank strongly but fail narrower threshold cells. The source-nominated 34 units contain effectively all final-MLP post-ReLU energy and span its rank-34 state. Inside that live state, native axes are substantially more assignment-legible than dimension-matched rotations and positive mixtures; state-level probes recover the three assignments without equally clean single axes.

The classifier uses this report continuously. Every live unit adds positive SAT-minus-UNSAT logit margin, while the threshold bits discard useful magnitude from misses and rule-inconsistent activations. Group ablations change more discrete decisions than the sum of individual ablations because additive evidence crosses an argmax margin jointly. No interaction term exists at this cut.

The same interpretation has a sharp operational limit: its bits help choose edits but cannot determine every legal edit response. Clean native axes therefore do not remove the need to specify what must be predicted or controlled, preserve graded information, compare like-dimensional representations and test the upstream route. The evidence justifies direct study of native mechanisms in this system and defines the seed, width and route experiments required before that strategy can carry weight in larger alignment settings.
'''
text = replace_between(text, "# Conclusion", "# Limitations and stopping decisions", conclusion)

old_limit = "The full-layer comparison is also conditional on the same checkpoint and bank. Its all-32 primary contains the sixteen assignments used to define the singleton cohort. The other sixteen are outside that singleton set but not outside source selection: post hoc inspection shows that every matched axis lies in the broader 34-unit report $J$, which the source nominated by direct output effect. The 63 rotations are a finite reference ensemble, not a population of learned alternative representations. Native post-ReLU coordinates are sparse and nonnegative, 442 units are dead on this bank, and the state has rank 34; generic Haar rotations are dense and signed, so an exchangeability null is false. Native loses on the weakest reusable assignment axis and on four differently structured controls. The zero-forward restriction and readout analyses were designed after opening the protected result. Individual and joint zero ablations are exact at the final affine cut but may be off-manifold, while within-stratum resampling preserves much of the semantic signal under test. These results establish a sparse decision-relevant report under declared comparisons, not prevalence, developmental ancestry, completeness or an upstream causal route."
new_limit = "The full-layer evidence is conditional on the same checkpoint and bank. The source selected $J$ by direct output effect before assigning rules, and $J$ contains effectively all post-ReLU energy here. The post hoc source-linkage, energy, live-subspace, positive-mixture, prevalence and probe analyses were designed after opening the protected full-space result. They correct its dimensional and geometric confounds but do not acquire protected status. Their 63-draw families are constructed references rather than exchangeability nulls or learned alternative representations. The 34-dimensional result is robust on the unique base formulas, but no bank-sampling interval or checkpoint population is available. Individual and grouped zero ablations are exact at the final affine cut and additive in logits, yet can be off-manifold; within-stratum resampling preserves much of the semantic signal under test. The result establishes a highly legible effective state and exact final-cut use, not prevalence, developmental ancestry, capacity-induced cleanliness, completeness or an upstream causal route."
if old_limit not in text:
    raise RuntimeError("full-layer limitation paragraph not found")
text = text.replace(old_limit, new_limit)

old_repro = "The zero-forward localisation diagnostic has prefix `2376633ac87df382`; the final sparse-report PDF figure has prefix `6c7c8ac6c6f16d4b`. The extraction covers all 512 units and preserves the earlier seventeen columns and logits bitwise. The PDF-build receipt records every full hash together with the freeze protocol, all cited figures and the diagnostic and figure scripts."
new_repro = "The zero-forward localisation diagnostic has prefix `2376633ac87df382`; the effective-state analysis v2 has prefix `b837c1447167432c` and binds its superseded indexing-error v1 by hash; the final sparse-report PDF figure has prefix `6c7c8ac6c6f16d4b`. The source notebook containing the unit-nomination rule has prefix `5410cc7cc28e28d3`. The extraction covers all 512 units and preserves the earlier seventeen columns and logits bitwise. The PDF-build receipt binds the protected SAT studies, manuscript, cited figures and named diagnostic scripts; older supporting components have their own manifests."
if old_repro not in text:
    raise RuntimeError("reproducibility sentence not found")
text = text.replace(old_repro, new_repro)

for banned in ("interacting report", "uses that report continuously and interactively", "Joint removal is also strongly superadditive", "spans every assignment rule", "6.6%", "clean existence case"):
    if banned in text:
        raise RuntimeError(f"stale phrase remains: {banned}")

PATH.write_text(text)
print(PATH)
