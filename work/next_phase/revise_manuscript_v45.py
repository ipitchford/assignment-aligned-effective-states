#!/usr/bin/env python3
"""Apply the v4.5 scientific and exposition revision with exact guards."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "outputs/causal-resolution-v3/paper/manuscript.md"
ABSTRACTS = ROOT / "outputs/causal-resolution-v3/paper/abstracts.md"
README = ROOT / "outputs/causal-resolution-v3/README.md"
STATUS = ROOT / "outputs/causal-resolution-v3/RESEARCH_STATUS.md"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = MANUSCRIPT.read_text()
    text = replace_once(text,
        'title: "A Sparse Native Assignment Report Spans the Effective Final-MLP State of a 2-SAT Transformer"\nsubtitle: "Protected label tests, geometry-matched basis controls and exact readout accounting"\ndate: "18 October 2026 · Version 4.4 working draft · Unrefereed"',
        'title: "Near-Atomic Assignment Structure in the Effective Final MLP of a 2-SAT Transformer"\nsubtitle: "Protected label tests, post hoc geometry controls and exact readout accounting"\ndate: "18 October 2026 · Version 4.5 working draft · Unrefereed"',
        "front matter")
    start = text.index("abstract: |\n")
    end = text.index("---\n\n# Introduction", start)
    abstract = """abstract: |
  Cammarata's *Neural Networks Are Already Clean* proposes that native neurons can be structured building blocks even when broader representations look distributed. We audit a width-512 final MLP in a five-variable 2-SAT transformer whose effective post-ReLU state on a protected 7,252-formula bank has rank 34 for 32 assignment atoms. The source published singleton-assignment labels for seventeen of those units. Those labels rank strongly (macro AUROC 0.991; unit range 0.961--1.000), but the frozen combined and threshold-fidelity criteria are not met: two units exceed a specificity limit by one of 32 cases, and four fail a positive cell. Retrospective accounting shows that the source-nominated 34 units contain effectively all bank energy. Every one is best matched to a single assignment at AUROC 0.946--1.000, covering 29 distinct assignments; orbit-grouped linear probes recover the three remaining assignments at AUROC 0.986--0.992. Retrospective native one-to-one matching remains above 63 dimension-matched rotations after raw, standardised and positive-mixture comparisons (native mean 0.984; best standardised rotation 0.914), while structured control labels reverse direction. The final readout exactly sums nonnegative SAT evidence from all 34 units. Replacing continuous activations by cross-fitted threshold bits preserves 98.39% of argmaxes but lowers task accuracy from 99.49% to 98.15%, directly measuring decision-relevant magnitude loss. Joint zeroing lowers accuracy to 20.06%, establishing that this final MLP is the sole SAT-evidence channel at the audited cut, not that assignment identity is read out. Nominal width therefore overstates the operative capacity of this checkpoint. The result is a precise single-checkpoint case in which an already-clean description substantially holds, together with the failed gates, three shared-axis exceptions and seed/width tests needed before it can guide claims about prevalence, development or larger alignment systems.
"""
    text = text[:start] + abstract + text[end:]

    text = replace_once(text,
        "The point of contact with *Already Clean* is bounded but direct. The final MLP has width 512 for a task with 32 assignments and 40 clause types. Palumbo et al. first selected 34 units whose largest direct effect on the SAT output exceeded 0.1, then derived logical rules for those units; seventeen rules name one satisfying assignment. This is a source-nominated causal report, not a random neuron sample. The design creates no demonstrated capacity pressure and cannot estimate how often arbitrary units are legible. Polysemanticity can depend on feature capacity (Scherlis et al., 2022) but can also arise incidentally despite ample capacity (Lecomte et al., 2024); neither origin is identified here. We test whether the fixed singleton labels rank fresh cases, survive the source threshold, support specified new uses, and sit inside a broader assignment report when the complete layer is inspected. A fixed-rotation comparison is retained as one protected diagnostic, but rotating a sparse nonnegative ReLU state makes dense signed coordinates, so native and rotated bases are not exchangeable under a generic null. The study therefore does not adjudicate recursive development, prevalence or capacity-driven superposition.",
        "The point of contact with *Already Clean* is bounded but direct. The final MLP has nominal width 512 for a task with 32 assignments and 40 clause types. Palumbo et al. selected 34 units whose largest direct effect on the SAT output exceeded 0.1, then derived logical rules for those units; seventeen rules name one satisfying assignment. This is a source-nominated causal report, not a random neuron sample. Complete-layer extraction changes the capacity description: on the audited bank the state has effective rank 34, close to the 32 assignment atoms, despite the nominal width. Whether regularisation, task structure or another training contingency caused that economy is unidentified. Polysemanticity can depend on feature capacity (Scherlis et al., 2022) and can also arise incidentally (Lecomte et al., 2024). We test whether the fixed singleton labels rank fresh cases, survive the source threshold, support specified new uses, and sit inside a broader assignment report when the complete layer is inspected. A fixed-rotation comparison is retained as one protected diagnostic, but rotating a sparse nonnegative ReLU state makes dense signed coordinates, so native and rotated bases are not exchangeable under a generic null. The study therefore does not adjudicate recursive development, prevalence or the causal origin of the effective-width collapse.",
        "capacity paragraph")
    text = replace_once(text, "The central SAT evidence has six stages.", "The central SAT evidence is organised by the execution-status table below.", "stage count")
    text = replace_once(text,
        "Capacity pressure is only one proposed source of polysemanticity. Scherlis et al. (2022) analyse feature capacity and show in a toy model that importance, sparsity, kurtosis and architecture affect which features receive monosemantic or shared capacity. Lecomte et al. (2024) show theoretically and experimentally that random initialisation, regularisation and noise can also produce incidental mixed selectivity even when neurons are plentiful. The SAT checkpoint's width and source-selected units therefore make it a poor clean-versus-superposition experiment. We nevertheless execute the narrower direct comparator suggested by this literature: one-to-one assignment matching across all 512 native coordinates versus fixed Haar rotations of the same complete layer state. Its interpretation is restricted by the sparse nonnegative geometry of the native ReLU basis and by a single checkpoint and bank.",
        "Capacity pressure is only one proposed source of polysemanticity. Scherlis et al. (2022) analyse feature capacity and show in a toy model that importance, sparsity, kurtosis and architecture affect which features receive monosemantic or shared capacity. Lecomte et al. (2024) show theoretically and experimentally that random initialisation, regularisation and noise can also produce incidental mixed selectivity. Small algorithmic networks can exhibit unusually clean mechanisms after weight-decay-driven cleanup (Nanda et al., 2023), yet related tasks also show arbitrary circuit details across seeds (Chughtai et al., 2023) and qualitatively different algorithms under small hyperparameter or initialisation changes (Zhong et al., 2023). Dead or bank-silent neurons are also common enough that nominal width is an unreliable capacity measure (Voita et al., 2023). The SAT checkpoint is therefore informative as one precisely audited regime, not as a clean-versus-superposition population experiment. We execute one-to-one assignment matching across all 512 native coordinates, then interpret it using the bank-effective state, dimension-matched rotations, standardised rotations, positive mixtures and basis-invariant probes. Every comparison remains limited by the sparse nonnegative geometry, source selection, one checkpoint and one engineered bank.",
        "capacity literature")

    text = text.replace("Thus every observed threshold false positive is local to a formula that has a nearby solution.", "Thus every threshold false positive in these balanced base cells is local to a formula that has a nearby solution; this sentence does not cover the 1,000 full-bank false positives recorded by the readout audit.")
    text = replace_once(text,
        "Against all 512 units, thirteen nominated units rank first for their label and four rank second: unit 81 is behind 334 for FTTTF, 261 behind 374 for TTFTT, 411 behind 201 for TTFFT, and 416 behind 55 for the shared TTTFF label.",
        "Using post-ReLU two-sided AUROC on the complete bank, thirteen nominated units rank first among all 512 for their label and four rank second: unit 81 is behind 334 for FTTTF, 261 behind 374 for TTFTT, 411 behind 201 for TTFFT, and 416 behind 55 for the shared TTTFF label. Under the protected preactivation positive-direction score, the 13/4 split is unchanged but unit 416 ranks first and unit 55 second for TTTFF; the scoring basis therefore accompanies every rank claim.",
        "rank basis")
    text = replace_once(text,
        "near cases form only 18.1% of fixed and 19.7% of switching negatives, versus 50% in the base bank.",
        "near cases form only 18.1% of all fixed negatives and 19.7% of all switching negatives, versus 33.3% of all base negatives. Among satisfiable negatives alone the corresponding shares are 27.2%, 29.5% and 50.0%.",
        "transposition denominator")
    text = replace_once(text,
        "The raw finite-reference ranks are 1/64 for both primaries; applying the frozen two-test Holm rule gives 1/32. These are extremeness ranks inside a fixed reference ensemble, not randomisation $p$ values, because sparse nonnegative native coordinates and dense signed rotations are not exchangeable.",
        "The raw finite-reference ranks are 1/64 for both primaries; the frozen two-test procedure transforms those ranks to 1/32. This is a pre-specified decision rule on finite-ensemble extremeness ranks, not a Holm familywise-error guarantee or a randomisation $p$ value, because sparse nonnegative native coordinates and dense signed rotations are not exchangeable. With 63 references the rank is saturated; native-minus-best margins carry the effect-size information.",
        "Holm meaning")
    text = replace_once(text,
        "The count descriptives in this protected comparison are partly mechanical.",
        "The protected control labels reverse direction: native versus Haar ranges are 0.543 versus 0.644--0.652 for clause-presence matching, 0.564 versus 0.785--0.929 for the SAT label, 0.534 versus 0.590--0.633 for sole-solution status, and 0.568 versus 0.762--0.901 for multi-solution status. SAT's reversal is expected from the dense positive readout direction across all live units.\n\nThe count descriptives in this protected comparison are partly mechanical.",
        "protected controls")
    text = replace_once(text,
        "Thus $J$ is the effective live state of this final MLP on the bank. Post hoc exclusion of $J$ collapses assignment matching because it removes essentially the entire state, not because alignment was localised against a substantial non-$J$ alternative.",
        "Thus $J$ is the effective live state of this final MLP on the bank. A zero-forward parameter audit adds an important qualification. Of the 478 non-$J$ units, 477 have a maximum input-norm, bias or output-norm scale at or below $3.74\\times10^{-12}$; unit 386 retains modest weights (input norm 0.0544, output norm 0.0848) but never activates on this bank. The effective-state identity is therefore almost entirely weight-level decay, with one bank-silent exception; it is not an input-independent theorem about every possible formula. Post hoc exclusion of $J$ collapses assignment matching because it removes essentially the entire observed state, not because alignment was localised against a substantial non-$J$ alternative.",
        "weight level audit")
    text = replace_once(text,
        "| 70-dimensional Haar rotations of numerically active columns | 0.924260 | 0.932969 | +0.051265 | $-0.011003$ |\n| Dense nonnegative mixtures of $J$ |",
        "| 70-dimensional Haar rotations of numerically active columns | 0.924260 | 0.932969 | +0.051265 | $-0.011003$ |\n| Standardised 34-dimensional Haar rotations | 0.902659 | 0.914323 | +0.069910 | +0.031780 |\n| Dense nonnegative mixtures of $J$ |",
        "standardised reference row")
    text = replace_once(text,
        "Within the exact 34-dimensional live span, even the weakest native matched axis exceeds every rotation and dense positive mixture.",
        "Within the exact 34-dimensional live span, the weakest native matched axis exceeds the weakest matched axis of every raw rotation, every standardised rotation and every dense positive mixture.",
        "weakest wording")
    text = replace_once(text,
        "Sparse native coordinates therefore do not generically win for equally rare labels or nearby clause-count events.",
        "Native coordinates therefore do not win on these noise or structured control labels. The shuffled-label reversal is partly mechanical because sparse post-ReLU axes create many average-rank ties and consequently less extreme noise AUROCs than dense continuous rotations; the structured clause and solution-count controls carry more semantic weight.",
        "control interpretation")
    text = replace_once(text,
        "An orbit-grouped five-fold linear probe on the 34-dimensional state reaches mean AUROC 0.9956 and minimum 0.9842.",
        "The basis-invariant headline is the native-to-probe gap. An orbit-grouped five-fold linear probe on the 34-dimensional state reaches mean AUROC 0.9956 and minimum 0.9842. For 29 of 32 assignments, one native unit is within 0.018 of that probe.",
        "probe headline")
    text = replace_once(text,
        "All 32 axes selected by the protected match belong to $J$",
        "A retrospective 32-label competition across all 34 source units finds that every unit is best matched to one assignment at AUROC 0.946--1.000, including all seventeen units whose published source rule is not singleton; collectively their best matches cover 29 assignments. All 32 axes selected by the protected match belong to $J$",
        "all 34 competition")
    text = replace_once(text,
        "Because $J$ is the whole effective final-MLP state, this is a layer-level ablation, not evidence that a small sub-bank uniquely carries the task.",
        "Because $J$ is the whole effective final-MLP state on the bank, this is a layer-level ablation showing that the final MLP is the sole SAT-evidence channel at this cut. The readout is assignment-agnostic: it sums positive evidence and does not itself use assignment identity.",
        "all34 meaning")
    text = replace_once(text,
        "The thresholded logical report is therefore a useful coarse description, while the affine readout uses continuous positive evidence.",
        "The thresholded logical report is therefore a useful coarse description, while the affine readout uses continuous positive evidence. A five-fold orbit-grouped substitution makes this loss operational: replacing each live activation by its threshold bit times a training-fold median preserves 98.39% of native argmaxes but lowers task accuracy from 99.49% to 98.15%; sole-solution accuracy falls from 98.79% to 95.52%. This is a cross-fitted diagnostic of this threshold code, not an optimum over all bit decoders.",
        "bit substitution")
    text = replace_once(text,
        "Adding the truth table leaves 1,484; these are properties of the edit target under the declared current-semantics context, not evidence about the interpretation. The refinement removes 1,000 errors, or 0.2504% of all 399,360 edit rows. Only this removed component diagnoses information absent from the published bits on the retained bank.",
        "The joint published-bit-plus-truth-table partition leaves 1,484 minimum errors. This is the joint floor, not a purely logical target error: truth table alone leaves about 2,100 errors, so the published bits remove about 616 errors beyond current satisfiability semantics, while truth-table refinement removes 1,000 errors beyond the bits. Only the latter 1,000, or 0.2504% of all 399,360 edit rows, diagnoses information absent from the published bits on the retained bank.",
        "edit floor")

    old_table = """| Proposed use | Available information | Result and limit |
|:--|:--|:--|
| Predict a legal edit from the current published report | 34 bits and the common action | Current truth does not determine the target. Adding it removes 1,000/399,360 empirical errors attributable to report information; 1,484 logical target errors remain. The report-specific rate is occupancy-dependent. |
| Select an UNSAT-producing edit | Initial report or post-ReLU magnitudes and one action menu | Published bits guide choices above action context alone. A one-seed post hoc factorial changes sign with action encoding, so no general bits-versus-magnitudes ordering follows. |
| Forecast a named unit after a paired edit | Fitted readiness bit and assignment-specific action effect | 90.95% balanced accuracy; the readiness bit adds 12.24 points over semantic action effect alone. This is an opened-bank predictive result. |
| Test all designated singleton-assignment labels | Seventeen fixed units, five semantic strata and the source threshold | All ranking gates pass. A specificity guard fails for two units and four fail a positive cell. All 36/544 threshold false positives occur on Hamming-one SAT negatives. |
| Characterise the complete final-MLP state | All 512 coordinates; protected full-space rotations; retrospective energy, in-span, positive-mixture, prevalence and probe controls | The source 34 span effectively the whole rank-34 state. Native matching exceeds every like-dimensional and nonnegative-mixture reference; rare shuffled and one-violation labels reverse direction. Three assignments are much better recovered by state-level probes than by one native axis. |
| Test selected labels on consequential matched edits | Four-direction protected pilot with fresh quartets | Three directions detect at least 31/32 positives at 0.5; unit 411 detects 24/32. Units 374 and 201 are same-assignment axes, so the activity-profile comparisons are not semantic nulls. |
| Account for the exact final readout | Complete state, exact individual and grouped zeroing, category-level margin accounting | The readout is additive. All 34 live units have positive SAT-minus-UNSAT coefficients; grouped removal lowers accuracy to 20.06%. Rule-inconsistent and sub-threshold activations can support correct decisions. |"""
    new_table = """| Proposed use | Evidence status | Available information | Result and limit |
|:--|:--|:--|:--|
| Predict a legal edit from the current published report | Opened-bank exact lookup audit | 34 bits, complete truth table and the common action | Joint conditioning leaves 1,484 minimum errors. Truth table removes 1,000 errors beyond bits; bits remove about 616 beyond truth alone. Rates are occupancy-dependent. |
| Select an UNSAT-producing edit | Opened-bank fitted pilot | Initial report or post-ReLU magnitudes and one action menu | Published bits guide choices above action context alone. A one-seed post hoc factorial changes sign with action encoding, so no general bits-versus-magnitudes ordering follows. |
| Forecast a named unit after a paired edit | Frozen contract on opened states | Fitted readiness bit and assignment-specific action effect | 90.95% balanced accuracy; the readiness bit adds 12.24 points over semantic action effect alone. This is an opened-bank predictive result. |
| Test all designated singleton-assignment labels | Protected bank and criteria; post hoc diagnostics separated | Seventeen fixed units, five semantic strata and the source threshold | All ranking gates pass. A specificity guard fails for two units and four fail a positive cell. The 36/544 base-cell false positives are Hamming-one SAT negatives. |
| Characterise the complete final-MLP state | Protected full-space comparison; retrospective geometry and probe controls | All 512 coordinates and the bank-effective 34-dimensional state | Native matching exceeds every raw and standardised dimension-matched rotation and positive-mixture reference. Probes recover three assignments lacking equally clean single axes. |
| Test selected labels on consequential matched edits | Protected four-direction pilot; post hoc causal diagnostics | Fresh matched quartets | Three directions detect at least 31/32 positives at 0.5; unit 411 detects 24/32. Units 374 and 201 are stronger same-assignment axes, so the activity-profile comparisons are not semantic nulls. |
| Account for the exact final readout | Retrospective exact arithmetic on retained states | Complete state, zeroing, category margins and cross-fitted threshold substitution | The readout sums positive SAT evidence. All-state removal lowers accuracy to 20.06%; threshold substitution preserves 98.39% of argmaxes and lowers accuracy to 98.15%. |"""
    text = replace_once(text, old_table, new_table, "status table")

    text = replace_once(text,
        "A native representation can be far more axis-legible than alternative bases inside the same effective state, while still losing information when converted to Boolean feature flags.",
        "A native representation can be far more axis-legible than random dimension-matched rotations and positive mixtures inside the same effective state, while still losing information when converted to Boolean feature flags.",
        "alignment comparator scope")
    text = replace_once(text,
        "A feature monitor that stores only threshold bits would hide contributions from sub-threshold and rule-inconsistent activations. For safety work, continuous feature values, readout coefficients and decision margins may matter even when the underlying axes are semantically clean.",
        "The cross-fitted threshold substitution quantifies the monitoring loss: it changes 1.61% of argmaxes and reduces sole-solution accuracy by 3.28 percentage points. For safety work, continuous feature values, readout coefficients and decision margins may matter even when the underlying axes are semantically clean; any compressed monitor should be tested end to end on its actual decision target.",
        "alignment bit claim")
    text = replace_once(text,
        "Widths below, near and above the 32 assignment atoms would reveal whether dedicated native axes persist under capacity pressure.",
        "Widths below, near and above the 32 assignment atoms would reveal whether the observed 34-unit effective state and dedicated native axes persist as operative capacity changes.",
        "sweep framing")
    text = replace_once(text,
        "That experiment directly tests the capacity account that this width-512 checkpoint cannot.",
        "That experiment directly tests whether effective width tracks the atom count and whether regularisation, nominal width or seed controls the three shared-axis exceptions.",
        "sweep conclusion")
    text = replace_once(text,
        "Every live unit adds positive SAT-minus-UNSAT logit margin, while the threshold bits discard useful magnitude from misses and rule-inconsistent activations. Group ablations change more discrete decisions than the sum of individual ablations because additive evidence crosses an argmax margin jointly. No interaction term exists at this cut.",
        "Every live unit adds positive SAT-minus-UNSAT logit margin. Cross-fitted threshold substitution preserves 98.39% of native argmaxes but reduces task accuracy from 99.49% to 98.15%, directly measuring useful discarded magnitude. For the seventeen-unit group, joint ablation changes more decisions than the sum of individual counts because additive evidence crosses an argmax margin jointly; individual pairs need not be superadditive. No interaction term exists at this cut.",
        "conclusion readout")
    text = replace_once(text,
        "Their 63-draw families are constructed references rather than exchangeability nulls or learned alternative representations.",
        "Their 63-draw families are constructed references rather than exchangeability nulls or learned alternative representations. Standardising each live coordinate before rotation preserves the native mean advantage but does not remove large native zero-ties; a learned-basis comparator remains open. The weight audit finds one non-report unit, 386, with nontrivial parameters but no activation on the bank, so the effective-state identity remains bank-conditional.",
        "limitations additions")
    text = replace_once(text,
        "the effective-state analysis v2 has prefix `b837c1447167432c` and binds its superseded indexing-error v1 by hash;",
        "the effective-state analysis v2 has prefix `b837c1447167432c` and binds its superseded indexing-error v1 by hash; the zero-forward weight, threshold-substitution, all-unit competition, task-error and standardised-rotation follow-up has prefix `4aeaaaf9be56882b`;",
        "repro binding")

    text = replace_once(text, "Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017).", "Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017).", "reference anchor")
    insert = """
Chughtai, B., Chan, L., & Nanda, N. (2023). A toy model of universality: Reverse engineering how networks learn group operations. *Proceedings of Machine Learning Research, 202*, 6243--6267. <https://proceedings.mlr.press/v202/chughtai23a.html>.

"""
    text = replace_once(text, "Chanin, D. (2026).", insert + "Chanin, D. (2026).", "Chughtai reference")
    nanda_ref = """
Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *International Conference on Learning Representations*. <https://openreview.net/forum?id=9XFSbDPmdW>.

"""
    text = replace_once(text, "Narang, H. (2026, July 12).", nanda_ref + "Narang, H. (2026, July 12).", "Nanda reference")
    voita_ref = """
Voita, E., Ferrando, J., & Nalmpantis, C. (2023). *Neurons in large language models: Dead, n-gram, positional*. <https://arxiv.org/abs/2309.04827>.

"""
    text = replace_once(text, "Voita, E., & Titov, I. (2020).", voita_ref + "Voita, E., & Titov, I. (2020).", "Voita reference")
    zhong_ref = """
Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The Clock and the Pizza: Two stories in mechanistic explanation of neural networks. *Advances in Neural Information Processing Systems, 36*. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/56cbfbf49937a0873d451343ddc8c57d-Abstract-Conference.html>.

"""
    text = replace_once(text, "Zhou, J., & Klindt, D. (2026).", zhong_ref + "Zhou, J., & Klindt, D. (2026).", "Zhong reference")
    if "Version 4.4" in text or "capacity-unconstrained" in text or "1,484 logical target errors" in text:
        raise RuntimeError("stale v4.4 framing remains")
    MANUSCRIPT.write_text(text)

    abstracts = ABSTRACTS.read_text()
    english_start = abstracts.index("## English\n") + len("## English\n")
    english_end = abstracts.index("\n## ", english_start)
    english = abstract.removeprefix("abstract: |\n").replace("  ", "", 1).strip()
    abstracts = abstracts[:english_start] + "\n" + english + "\n" + abstracts[english_end:]
    abstracts = abstracts.replace("Version 4.4", "Version 4.5")
    ABSTRACTS.write_text(abstracts)

    readme = README.read_text().replace("Version 4.4", "Version 4.5")
    readme = readme.replace(
        "Exact readout accounting shows additive positive SAT evidence whose Boolean thresholding loses useful magnitude.",
        "Exact readout accounting shows additive positive SAT evidence; cross-fitted threshold substitution preserves 98.39% of native argmaxes while lowering task accuracy from 99.49% to 98.15%.")
    README.write_text(readme)

    status = STATUS.read_text().replace("**Version 4.4 is", "**Version 4.5 is")
    status = status.replace(
        "These analyses were designed after the protected result and do not change its decision.",
        "These analyses were designed after the protected result and do not change its decision. A standardised-coordinate sensitivity gives native 0.984 versus a best rotation of 0.914. Every one of the 34 units is best matched to one assignment at AUROC 0.946--1.000, collectively covering 29 assignments.")
    status = status.replace(
        "Source-threshold misses and false positives can still support\ncorrect decisions, establishing graded magnitude use.",
        "Source-threshold misses and false positives can still support\ncorrect decisions. Cross-fitted threshold substitution preserves 98.39% of native\nargmaxes while lowering task accuracy from 99.49% to 98.15%, directly measuring\ngraded magnitude use.")
    STATUS.write_text(status)

    print("updated", MANUSCRIPT, ABSTRACTS, README, STATUS)


if __name__ == "__main__":
    main()
