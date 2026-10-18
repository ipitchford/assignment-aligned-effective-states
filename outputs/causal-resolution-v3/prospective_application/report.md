# Predicting actions from the original report

This exploratory study asks whether a report computed **before an intervention** suffices to predict its effects. It reuses all six trained checkpoints and the previously inspected digit splits. No neural model is retrained. “Prospective” describes the timing of the prediction query; this study is retrospective and is not preregistered or evaluated on a new pristine holdout.

## A stricter information interface

The previous experiments supplied both T(h) and T(h′) to the response predictor. Here the report encoder sees only h. A query supplies the fixed group action and attenuation strength, and the decoder must predict F(h′)−F(h) without seeing or computing the true post-action state h′. The four groups are the same fixed eight-unit subsets used previously. All four action effects are evaluated for each image.

We retain native8, native12, PCA8 and the nonlinear model’s random8 report maps. The shallow random8 maps are a new exploratory control: 32 seeded Gaussian candidate directions receive the same original retrospective discovery selection procedure as the other report families. We also encode the current centered output through the existing box quantiser: this costs 77 state bits for each shallow model, and 89, 89 and 88 bits for the three nonlinear models. It still sees only the original state. Computing current outputs in the encoder is explicitly charged; it is not counted as a semantic explanation.

A common prospective decoder has the fixed form s B_g[1,z]+s(1−s) G_g[1,z], where z is a standardized report, g is the group and s the strength. This is affine in the report and quadratic in strength. It is trained at strengths 1/2 and 1 on discovery-fit images; ridge is chosen on the original separate discovery-validation images. It is evaluated at those strengths and at 1/4 and 3/4. Standardization uses discovery-fit codes only. The polynomial is a predetermined limited model, not an exact assumption about the nonlinear classifier.

This decoder has 80(d+1) coefficients for a d-coordinate report: 720 for native8/PCA8/random8, 1,040 for native12, 880 for the ten-coordinate current-output quantiser, and 80 for the action-only baseline. This is substantially more permissive than the earlier shared report-difference decoder. Errors should not be compared across those studies as if only the information timing had changed.

A single query additionally requires 4 action bits to specify one of four groups and four strengths. Native8 therefore uses 12 query bits; the shallow current-output code uses 81. The margin-action decision described below supplies a further ordered top/runner class pair, at most 7 bits. No exact current margin, image identity, digit label, or true post-action report is supplied. The current-output encoder uses the known downstream computation (320 centered head coefficients for the shallow model, 698 weights/biases for the nonlinear model), in addition to scalar quantiser constants. Dense PCA/random8 needs at most 256 projection coefficients; native8 uses eight coordinate predicates.

## Measured response prediction

MSE sums squared error over ten centered logits, then averages the 2,912 image/action rows in each bank. Images within a model and their repeated interventions are dependent. These are descriptive observations on the reused split.

| Seed | Downstream readout | Native8 unseen MSE | Native12 | PCA8 | Random8 | Current output | Action only |
|---|---|---:|---:|---:|---:|---:|---:|
|17|linear|5.631|4.555|9.663|7.687|0.682|14.397|
|29|linear|5.829|4.553|12.510|6.306|0.845|13.085|
|43|linear|5.882|4.296|14.449|7.482|0.569|15.478|
|117|nonlinear|12.782|10.041|20.587|13.077|3.481|26.848|
|129|nonlinear|10.933|8.814|15.305|11.955|2.696|24.474|
|143|nonlinear|13.096|11.197|25.824|15.983|3.413|31.705|

Current-output reports predict these natural-image action effects substantially better than the tested low-bit reports, but use more report bits and an encoder that runs the downstream computation. Their maximum unseen-strength errors are 1.837, 2.282, 1.601, 4.492, 4.086, 3.888 in seed order 17,29,43,117,129,143, so this fitted decoder does not meet epsilon1. This fitted failure does not itself prove an information obstruction: all 364 current-output codes are distinct on each observed natural-image bank, and the finite lookup oracle has zero error there.

The separate exact action-closure analysis supplies the stronger structural statement for the shallow models: the current centered output has rank 9, whereas a linear state summary sufficient for all four group effects must have rank at least 32. That analysis constructs admissible continuous input pairs with identical current output and different same-action responses. Those exact synthetic witnesses are distinct from the natural-image decision witnesses below; neither should be represented as a natural-data prevalence estimate.

The retained native-coordinate precision codes provide a costed prospective numerical control. Their decoder reconstructs an approximate original state from the code, applies the action to that reconstruction, and evaluates the known downstream model. It never accesses the true future state. The existing global bounds remain sufficient, and every evaluated response passes epsilon1 by exact arithmetic. This is an explicit numerical simulator using 212–287 state bits and the original downstream computation, not a semantic mechanism discovered by the experiment.

## Natural-image decision failures

For a fixed 75% attenuation, define the action’s value as its decrease in the original top-class-versus-runner-up margin. Given only the report and the two class labels, choose one of the four groups to maximize that decrease. This is a concrete targeted intervention question; it is not equivalent to finding any intervention that changes the full ten-way predicted class.

Two images with the same report and the same ordered class pair force any deterministic report-based policy to choose the same group. Let R_ig be the regret of choosing group g on image i: the best attainable margin drop on that image minus the chosen group’s drop. For a report/context fibre, the exact smallest possible worst-case regret is min_g max_i R_ig. The empirical mean-regret floor is obtained by minimizing the sum in each fibre. Both quantities are computed by exhaustive enumeration of four actions using integer response numerators. The worst-case values below concern deterministic policies; a randomized minimax policy defines a different optimization.

| Seed | Native8 minimum worst-case regret | Empirical minimum mean regret | Fitted policy action accuracy | Fitted mean regret |
|---|---:|---:|---:|---:|
|17|1.7225|0.0184|84.34%|0.1758|
|29|2.4091|0.0256|83.24%|0.1817|
|43|3.3756|0.0452|66.48%|0.4075|
|117|1.8672|0.0191|84.89%|0.2944|
|129|3.4236|0.0776|79.67%|0.3900|
|143|4.8722|0.0774|77.75%|0.3999|

Each worst-case native8 bound is attained by a two-image certificate with identical report/context and different unique best actions. The mean floors are much smaller than the tail bounds. These examples demonstrate consequential possible failures on the finite bank; they do not establish that such failures dominate average behaviour. The empirical optimal lookup policies use observed outcomes and are diagnostic oracles, not deployed explainers.

## Fixed classification illustrations

The following examples were selected descriptively from the exact native8 certificates. Their 75% action strength and top/runner-margin objective were fixed before this illustration; no action strength or margin threshold was searched to manufacture a classification change.

**Shallow seed 43: a classification-changing action must differ.** Two naturally occurring digit 8 images, 1305 and 253, have the same native8 code and the same predicted class 8 and runner-up 1. Their original margins are 5.17655 and 4.53109. Only action 2 changes the first prediction 8→1; only action 3 changes the second 8→1. Any deterministic policy restricted to the common report/context therefore fails to select a class-flipping action for at least one of these images, although each has such an action. Under a uniform distribution on this particular pair, its success rate is at most 1/2. This is a pair-specific claim, not the test-set failure rate.

| Seed 43 image | Original 8−1 margin | After action 0 | After action 1 | After action 2 | After action 3 | Full predicted classes after 0–3 |
|---|---:|---:|---:|---:|---:|---|
|1305|5.17655|6.37360|9.04904|-0.86463|2.51096|[8, 8, 1, 8]|
|253|4.53109|6.41540|7.18880|2.47000|-1.10297|[8, 8, 8, 1]|

**Nonlinear seed 143: the largest margin-regret certificate is target-specific.** Images 1120 and 11 are both labelled 1, with common predicted class 1 and runner-up 8. Their shared report is 11011010. The unique best groups for reducing the 1−8 margin are 1 and 3, and every common deterministic action incurs at least 4.872248814 margin-regret on one image. However, action 1 changes the full prediction of both images, to 7 and 3 respectively. The different best margin actions therefore do not establish different necessary class-flip actions in this pair. Other classes matter.

| Seed 143 image | Original 1−8 margin | After action 0 | After action 1 | After action 2 | After action 3 | Full predicted classes after 0–3 |
|---|---:|---:|---:|---:|---:|---|
|1120|8.96845|13.90349|-1.48314|6.37514|8.88521|[1, 7, 1, 1]|
|11|12.57061|15.58473|8.44671|11.53904|3.57446|[1, 3, 1, 1]|

The figures display the images, all four exact margin drops, the actual predicted class after each action, and the finite-pair certificate. No neuron is assigned an unsupported semantic label such as a particular stroke detector.

## Evidence limits and reproduction

The audit repairs an important interface loophole by preventing the decoder from inspecting the actual post-action report. It also translates some report failures into a specified intervention decision on real images. Nevertheless, this remains six small models on one reused dataset. Models, action groups and hypotheses were not independently replicated; neither human interpretability nor larger-network coverage was tested. A good response simulator can move substantial computation into its encoder or decoder. All such costs are therefore retained.

The independent same-team integer check passed 297 checks and replayed all 34,944 response rows. It rebuilt every saved fitted prediction from the original code and action only, replayed precision predictions from approximate state only, and exhaustively rechecked every mean/worst decision-regret certificate. The files audit.py, decision_certificates.py, verify.py, results.json, decision_certificates.json and classification_witnesses.json contain the analysis and exact receipts. All source checkpoints and original reports are referenced in the sibling application and nonlinear_application directories, which must remain bundled.

The study does not by itself establish field-leading significance. Its contribution is a scoped, executable failure-and-decision test, with explicit separation of empirical prediction, exact finite obstruction, reachable synthetic aliases and wider generalization.
