#!/usr/bin/env python3
"""Apply the v4.6 final scientific revision with exact replacement guards."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "outputs/causal-resolution-v3/paper/manuscript.md"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = MANUSCRIPT.read_text()
    text = replace_once(
        text,
        'title: "Near-Atomic Assignment Structure in the Effective Final MLP of a 2-SAT Transformer"',
        'title: "Assignment-Aligned Effective States in a 2-SAT Transformer"',
        "title",
    )
    text = replace_once(
        text,
        'subtitle: "Protected label tests, post hoc geometry controls and exact readout accounting"',
        'subtitle: "Protected label tests, learned-basis controls and exact readout accounting"',
        "subtitle",
    )
    text = replace_once(
        text,
        'date: "18 October 2026 · Version 4.5 working draft · Unrefereed"',
        'date: "18 October 2026 · Version 4.6 working draft · Unrefereed"',
        "version",
    )

    start = text.index("abstract: |\n")
    end = text.index("\n---\n", start)
    abstract = """abstract: |
  Cammarata's *Neural Networks Are Already Clean* proposes that native neurons can be structured building blocks even when broader representations look distributed. We test one published interpretation of a width-512 final MLP in a five-variable 2-SAT transformer. On a protected 7,252-formula bank, seventeen source-nominated singleton-assignment labels rank strongly (macro AUROC 0.991; unit range 0.961--1.000), but the frozen combined and threshold-fidelity criteria are not met: two units exceed a specificity limit by one of 32 cases, and four fail a positive cell. A separately protected complete-layer comparison places native assignment matching above 63 fixed full-space Haar rotations, although sparse nonnegative native coordinates and dense rotations are not exchangeable. Every further structural claim is retrospective. The source-nominated 34 units form the entire effective post-ReLU state on this bank. Each unit's best assignment AUROC is 0.946--1.000, covering 29 assignments, while activation-mass purity ranges from 0.595 to 0.994. Orbit-grouped probes recover all 32 assignments. Native matching exceeds raw and standardised rotations and fixed nonnegative mixtures, but 34-component nonnegative matrix factorisation slightly exceeds the native matching mean (0.986 versus 0.984) while recovering 32 of 34 native axes above cosine 0.90. The state is therefore strongly assignment-aligned and close to a natural nonnegative basis, without making the native coordinates uniquely optimal. The exact readout sums positive SAT evidence. All-state zeroing gives 20.06% accuracy, approximately the 20.04% UNSAT prevalence. Fixed-readout threshold substitution gives 98.15% accuracy; an orbit-grouped refitted bit classifier reaches 99.12%, versus 99.68% for refitted continuous activations. This single abundant-capacity regime is predicted by both Already Clean and superposition accounts and does not discriminate between them. Its contribution is a status-aware audit that separates label ranking, threshold fidelity, effective-state geometry, learned alternatives, causal use and action sufficiency, and specifies the seed, width and upstream-route experiments needed for broader alignment claims."""
    text = text[:start] + abstract + text[end:]

    text = replace_once(
        text,
        "The point of contact with *Already Clean* is bounded but direct. The final MLP has nominal width 512 for a task with 32 assignments and 40 clause types. Palumbo et al. selected 34 units whose largest direct effect on the SAT output exceeded 0.1, then derived logical rules for those units; seventeen rules name one satisfying assignment. This is a source-nominated causal report, not a random neuron sample. Complete-layer extraction changes the capacity description: on the audited bank the state has effective rank 34, close to the 32 assignment atoms, despite the nominal width. Whether regularisation, task structure or another training contingency caused that economy is unidentified. Polysemanticity can depend on feature capacity (Scherlis et al., 2022) and can also arise incidentally (Lecomte et al., 2024). We test whether the fixed singleton labels rank fresh cases, survive the source threshold, support specified new uses, and sit inside a broader assignment report when the complete layer is inspected. A fixed-rotation comparison is retained as one protected diagnostic, but rotating a sparse nonnegative ReLU state makes dense signed coordinates, so native and rotated bases are not exchangeable under a generic null. The study therefore does not adjudicate recursive development, prevalence or the causal origin of the effective-width collapse.",
        "The point of contact with *Already Clean* is bounded but direct. The final MLP has nominal width 512 for a task with 32 assignments and 40 clause types. Palumbo et al. selected 34 units whose largest direct effect on the SAT output exceeded 0.1, then derived logical rules for those units; seventeen rules name one satisfying assignment. This is a source-nominated causal report, not a random neuron sample. Complete-layer extraction changes the capacity description: on the audited bank the state has effective rank 34 despite the nominal width. Whether weight decay, task structure or another training contingency caused that economy is unidentified. This regime is compatible with Cammarata's emphasis on native structure, but it is also predicted by superposition accounts when live capacity is abundant relative to the represented features; the checkpoint cannot discriminate those explanations. Polysemanticity can depend on feature capacity (Scherlis et al., 2022) and can also arise incidentally (Lecomte et al., 2024). We test whether the fixed singleton labels rank fresh cases, survive the source threshold, support specified new uses, and sit inside a broader assignment report when the complete layer is inspected. A fixed-rotation comparison is retained as one protected diagnostic, but rotating a sparse nonnegative ReLU state makes dense signed coordinates, so native and rotated bases are not exchangeable under a generic null. Learned decompositions are therefore evaluated retrospectively. The study does not adjudicate recursive development, prevalence or the causal origin of the effective-width collapse.",
        "introduction capacity framing",
    )
    text = replace_once(
        text,
        "| The effective final-MLP state is a sparse native assignment report | Complete 512-unit extraction; protected matching against 63 fixed full-space rotations; retrospective rank, energy, live-subspace, positive-mixture, prevalence and linear-probe controls | The source-nominated 34 units contain effectively all post-ReLU energy and span the rank-34 state. Native assignment matching remains above every 34-dimensional rotation and positive-mixture reference, while state-level linear probes recover the three weak single-axis assignments. This is one checkpoint and bank, with source selection fully disclosed |",
        "| The effective final-MLP state is strongly assignment-aligned | Complete 512-unit extraction; protected matching against 63 fixed full-space rotations; retrospective rank, purity, learned-basis, prevalence and linear-probe controls | The source-nominated 34 units form the bank-effective rank-34 state. Native matching exceeds fixed rotations and mixtures, but learned 34-component NMF slightly exceeds its mean while recovering 32 native axes. This is one checkpoint and engineered bank, with source selection fully disclosed |",
        "intro result table",
    )
    text = replace_once(
        text,
        "## The effective final-MLP state is a sparse native assignment report",
        "## Complete-layer assignment alignment and effective width",
        "section heading",
    )
    text = replace_once(
        text,
        "Palumbo et al.'s report $J$ contains 34 final-MLP units selected because each unit's largest direct effect on the SAT output exceeded 0.1 on the source training data. The source then fitted shallow logical rules to those units.",
        "Palumbo et al.'s report $J$ contains 34 final-MLP units selected because each unit's largest direct effect on the SAT output exceeded 0.1 on the source training data. Weight decay leaves 477 of the 478 units outside $J$ at parameter scale at most $3.74\\times10^{-12}$; the remaining unit is silent on this bank. Localisation to $J$ is therefore largely a mechanical consequence of the source's effect threshold and the trained weight pattern, not evidence that a selector isolated 34 units from a comparably energetic remainder. The source then fitted shallow logical rules to those units.",
        "mechanical localisation",
    )
    text = replace_once(
        text,
        "Its 34 columns contain $1-2\\times10^{-16}$ of the total post-ReLU squared norm; the other 36 numerically nonzero columns contribute only $6.98\\times10^{-53}$.",
        "Its 34 columns contain a squared-norm share of 0.9999999999999998 at reported precision; the other 36 numerically nonzero columns contribute a share of $6.98\\times10^{-53}$.",
        "energy precision",
    )
    text = replace_once(
        text,
        "The native mean exceeds every reference in all four families. Within the exact 34-dimensional live span, the weakest native matched axis exceeds the weakest matched axis of every raw rotation, every standardised rotation and every dense positive mixture.",
        "The native mean exceeds every reference in all five families. Within the exact 34-dimensional live span, the weakest native matched axis exceeds the weakest matched axis of every raw rotation, every standardised rotation and every dense positive mixture for the all-32 match. For the uncovered-assignment subset, one standardised rotation has minimum 0.8843 versus the native 0.8809; the mean advantage is not a universal per-axis advantage.",
        "reference family count and exception",
    )
    text = replace_once(
        text,
        "For the 32 labels $q_a=1$, meaning that assignment $a$ violates exactly one clause, native mean matching is 0.5530 and again loses to every rotation (0.6032--0.6151). Native coordinates therefore do not win on these noise or structured control labels.",
        "For the 32 labels $q_a=1$, meaning that assignment $a$ violates exactly one clause, native mean matching is 0.5530 and again loses to every rotation (0.6032--0.6151). Clause-violation count is semantically related to the source rules, so this is a monotonic-band sensitivity rather than a semantic null. Native coordinates therefore do not win on the shuffled-label control or this broader one-violation band.",
        "one violation control",
    )
    text = replace_once(
        text,
        "The basis-invariant headline is the native-to-probe gap. An orbit-grouped five-fold linear probe on the 34-dimensional state reaches mean AUROC 0.9956 and minimum 0.9842. For 29 of 32 assignments, one native unit is within 0.018 of that probe.",
        "A supervised learned comparator supplies an upper descriptive benchmark. An orbit-grouped five-fold linear probe on the 34-dimensional state reaches mean AUROC 0.9956 and minimum 0.9842, outperforming the native single-axis match. For 29 of 32 assignments, one native unit is within 0.018 of that probe.",
        "probe framing",
    )
    text = replace_once(
        text,
        "A retrospective 32-label competition across all 34 source units finds that every unit is best matched to one assignment at AUROC 0.946--1.000, including all seventeen units whose published source rule is not singleton; collectively their best matches cover 29 assignments. All 32 axes selected by the protected match belong to $J$",
        "A retrospective 32-label competition across all 34 source units finds that every unit's best assignment AUROC is 0.946--1.000, including all seventeen units whose published source rule is not singleton; collectively their best matches cover 29 assignments. Ranking is not purity: the fraction of each unit's summed activation mass on its best label ranges from 0.595 to 0.994 (median 0.893), with eighteen units below 0.90. The native axes are therefore graded local-region SAT-evidence directions rather than uniformly atomic assignment detectors. All 32 axes selected by the protected match belong to $J$",
        "all 34 purity",
    )
    anchor = "These source-linkage facts are retrospective consistency checks, not held-out discovery."
    addition = anchor + "\n\nLearned decompositions further limit the native-basis claim. Across five seeds, 34-component NMF attains mean one-to-one assignment matching 0.9862--0.9865, slightly above the native 0.9842; its weakest match ranges from 0.8807 to 0.8879, compared with native 0.8809. NMF recovers 32 of 34 native axes above absolute cosine 0.90, with mean matched cosine 0.9626. A 32-component NMF loses one assignment badly (minimum 0.5416), while 34-component ICA and the fixed sparse-dictionary sensitivity score below native. These post hoc methods were not given a protected common complexity budget, so their ordering is descriptive. The substantive result is an assignment-aligned nonnegative state whose native coordinates are largely recovered by NMF, not a uniquely optimal native basis."
    text = replace_once(text, anchor, addition, "learned basis paragraph")

    text = replace_once(
        text,
        "Jointly zeroing all 34 changes 5,764 decisions and reduces accuracy to 20.06%. Among baseline-correct cases",
        "Jointly zeroing all 34 changes 5,764 decisions and reduces accuracy to 20.06%, approximately the bank's 20.04% UNSAT prevalence. The residual path predicts UNSAT on 7,250/7,252 rows and produces no non-answer argmax. The native state is exactly zero on 992/1,453 UNSAT rows and on no SAT row, so the residual result mainly recovers the class prior-like UNSAT default rather than a second solver. Among baseline-correct cases",
        "joint zero detail",
    )
    text = replace_once(
        text,
        "A five-fold orbit-grouped substitution makes this loss operational: replacing each live activation by its threshold bit times a training-fold median preserves 98.39% of native argmaxes but lowers task accuracy from 99.49% to 98.15%; sole-solution accuracy falls from 98.79% to 95.52%. This is a cross-fitted diagnostic of this threshold code, not an optimum over all bit decoders.",
        "A five-fold orbit-grouped fixed-readout substitution makes the calibration loss operational: replacing each live activation by its threshold bit times a training-fold median preserves 98.39% of native argmaxes but lowers task accuracy from 99.49% to 98.15%; sole-solution accuracy falls from 98.79% to 95.52%, while UNSAT accuracy rises to 100%. This unrefitted native-readout substitution is an upper bound on the decision loss attributable to thresholding under that calibration. A separately refitted orbit-grouped logistic classifier reaches 99.12% from the 34 bits and 99.68% from continuous activations. Thresholding therefore leaves a smaller predictive gap after recalibration; neither supervised comparator is a causal abstraction or an optimum over all decoders.",
        "threshold decomposition",
    )
    text = replace_once(
        text,
        "The much lower task-error base rate and repeated use of each formula still make these counts descriptive; they do not establish that threshold errors are causally harmless.",
        "The much lower task-error base rate and repeated use of each formula still make these counts descriptive; they do not establish that threshold errors are causally harmless. A later task-error localisation finds 9/147 errors (6.12%; exact binomial 95% interval 2.84--11.30%) on the three weakest single-axis assignments, versus 26/2,753 (0.94%; 0.62--1.38%) on the other sole-solution assignments. Assignment composition is confounded, and FTTFF has 4/45 errors despite a strong dedicated axis, so this is an exploratory association rather than a causal error explanation.",
        "task error localisation",
    )

    old_implication = """The decision-relevant lesson is methodological. A native representation can be far more axis-legible than random dimension-matched rotations and positive mixtures inside the same effective state, while still losing information when converted to Boolean feature flags. In this checkpoint, the source 34 are not a sparse island in a 512-dimensional live layer; they are the entire effective final-MLP state on the bank. Within that state, however, native axes outperform dimension-matched rotations and positive mixtures on assignment labels, and the advantage disappears for prevalence-matched shuffled labels and one-violation controls. This is a sharper result than the original full-space rotation comparison and a more limited one than a claim about frontier models.

The exact readout gives a second lesson. The layer is linear at the intervention cut, yet joint removals produce nonlinear changes in discrete decisions because redundant positive evidence crosses argmax margins. The cross-fitted threshold substitution quantifies the monitoring loss: it changes 1.61% of argmaxes and reduces sole-solution accuracy by 3.28 percentage points. For safety work, continuous feature values, readout coefficients and decision margins may matter even when the underlying axes are semantically clean; any compressed monitor should be tested end to end on its actual decision target."""
    new_implication = """The decision-relevant lesson is methodological. In this checkpoint, the source 34 are the entire effective final-MLP state on the bank. Native axes are far more assignment-legible than random dimension-matched rotations and fixed positive mixtures, but learned NMF slightly improves the assignment match while largely recovering the native axes. This is evidence for a stable assignment-aligned nonnegative geometry, not for unique native optimality. Because live dimension is abundant relative to 32 assignment labels, both Already Clean and capacity-based superposition accounts predict a comparatively aligned regime. Widths below the atom count are required to separate them.

The exact readout gives a second lesson. The layer is linear at the intervention cut, yet joint removals produce nonlinear decision changes because redundant positive evidence crosses argmax margins. Fixed-readout threshold substitution changes 1.61% of argmaxes, but refitting the bits recovers 99.12% task accuracy. The remaining continuous-versus-bit classifier gap is 0.56 percentage points on this bank. For safety work, continuous feature values, readout calibration and decision margins can matter even when axes are semantically legible; any compressed monitor should be fitted and tested end to end on its actual decision target."""
    text = replace_once(text, old_implication, new_implication, "alignment implications")
    text = replace_once(
        text,
        "That experiment directly tests whether effective width tracks the atom count and whether regularisation, nominal width or seed controls the three shared-axis exceptions.",
        "Widths below 32 are the decisive cells for distinguishing an Already Clean account from a capacity-based superposition account; the remaining cells test whether effective width tracks the atom count and whether regularisation, nominal width or seed controls the shared-axis exceptions.",
        "decisive sweep",
    )

    old_conclusion = """Cammarata's proposal becomes testable when “clean” is divided into label ranking, threshold fidelity, basis legibility, causal use and action sufficiency. In this checkpoint, published singleton labels rank strongly but fail narrower threshold cells. The source-nominated 34 units contain effectively all final-MLP post-ReLU energy and span its rank-34 state. Inside that live state, native axes are substantially more assignment-legible than dimension-matched rotations and positive mixtures; state-level probes recover the three assignments without equally clean single axes.

The classifier uses this report continuously. Every live unit adds positive SAT-minus-UNSAT logit margin. Cross-fitted threshold substitution preserves 98.39% of native argmaxes but reduces task accuracy from 99.49% to 98.15%, directly measuring useful discarded magnitude. For the seventeen-unit group, joint ablation changes more decisions than the sum of individual counts because additive evidence crosses an argmax margin jointly; individual pairs need not be superadditive. No interaction term exists at this cut.

The same interpretation has a sharp operational limit: its bits help choose edits but cannot determine every legal edit response. Clean native axes therefore do not remove the need to specify what must be predicted or controlled, preserve graded information, compare like-dimensional representations and test the upstream route. The evidence justifies direct study of native mechanisms in this system and defines the seed, width and route experiments required before that strategy can carry weight in larger alignment settings."""
    new_conclusion = """Cammarata's proposal becomes testable when “clean” is divided into label ranking, threshold fidelity, effective-state geometry, causal use and action sufficiency. In this checkpoint, published singleton labels rank strongly but fail narrower threshold cells. The source-nominated 34 units form the rank-34 bank-effective state. Their axes are substantially more assignment-legible than random dimension-matched rotations and fixed positive mixtures, but activation-mass purity is heterogeneous and learned NMF slightly improves matching while recovering most native axes. The defensible description is a graded, assignment-aligned nonnegative state with strong native resemblance.

The classifier uses this state continuously. Every live unit adds positive SAT-minus-UNSAT logit margin. Fixed-readout threshold substitution preserves 98.39% of native argmaxes and gives 98.15% task accuracy; refitted bit and continuous classifiers reach 99.12% and 99.68%. For the seventeen-unit group, joint ablation changes more decisions than the sum of individual counts because additive evidence crosses an argmax margin jointly; no interaction term exists at this cut.

The same interpretation has a sharp operational limit: its bits help choose edits but cannot determine every legal edit response. This abundant-capacity checkpoint is consistent with both Already Clean and superposition accounts. The evidence supports direct study of its native mechanism while making no claim about systems with more features than live units. Effective-state accounting, learned-basis comparison, preserved graded information, and upstream interventions are required before the strategy can carry weight in larger alignment settings."""
    text = replace_once(text, old_conclusion, new_conclusion, "conclusion")

    old_limit = "The full-layer evidence is conditional on the same checkpoint and bank. The source selected $J$ by direct output effect before assigning rules, and $J$ contains effectively all post-ReLU energy here. The post hoc source-linkage, energy, live-subspace, positive-mixture, prevalence and probe analyses were designed after opening the protected full-space result. They correct its dimensional and geometric confounds but do not acquire protected status. Their 63-draw families are constructed references rather than exchangeability nulls or learned alternative representations. Standardising each live coordinate before rotation preserves the native mean advantage but does not remove large native zero-ties; a learned-basis comparator remains open. The weight audit finds one non-report unit, 386, with nontrivial parameters but no activation on the bank, so the effective-state identity remains bank-conditional."
    new_limit = "The full-layer evidence is conditional on the same checkpoint and bank. The source selected $J$ by direct output effect before assigning rules, and weight decay makes $J$ effectively the whole post-ReLU state here. The post hoc source-linkage, energy, live-subspace, positive-mixture, prevalence, probe, purity, NMF, ICA, sparse-dictionary, error-localisation and refitted-readout analyses were designed after opening the protected full-space result. They correct or expose dimensional and geometric confounds but do not acquire protected status. The 63-draw families are constructed references rather than exchangeability nulls. The learned methods were not assigned a protected common complexity budget, and their five seeds do not estimate model-population uncertainty. Standardising each live coordinate before rotation preserves the native mean advantage but does not remove large native zero-ties. The weight audit finds one non-report unit, 386, with nontrivial parameters but no activation on the bank, so the effective-state identity remains bank-conditional."
    text = replace_once(text, old_limit, new_limit, "full layer limitations")
    text = replace_once(
        text,
        "Its preflight and self-checks passed 3,227 assertions before native execution, and a full-length Opus 5.5 model-generated static review returned GO.",
        "Its preflight and self-checks passed 3,227 assertions before native execution.",
        "remove review verdict",
    )
    text = replace_once(
        text,
        "the zero-forward weight, threshold-substitution, all-unit competition, task-error and standardised-rotation follow-up has prefix `4aeaaaf9be56882b`;",
        "the zero-forward weight, threshold-substitution, all-unit competition, task-error and standardised-rotation follow-up has prefix `4aeaaaf9be56882b`; the zero-forward purity, learned-basis, refitted-readout and joint-zero follow-up has prefix `25a1ab870aa87701`;",
        "v5 provenance",
    )
    text = text.replace("Version 4.5", "Version 4.6")
    banned = ("Near-Atomic", "all four families", "monitoring loss", "a learned-basis comparator remains open")
    for phrase in banned:
        if phrase in text:
            raise RuntimeError(f"stale v4.5 phrase remains: {phrase}")
    MANUSCRIPT.write_text(text)
    print("updated", MANUSCRIPT)


if __name__ == "__main__":
    main()
