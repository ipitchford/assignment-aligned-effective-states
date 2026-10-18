# Exploratory nonlinear-downstream architecture extension

This supplement tests the first hidden-state cut of a 64→32 ReLU→16 ReLU→10 classifier. It was designed after the shallow-model study and reuses the same already-inspected image split. Its choices were fixed before evaluating these new models on that split, but it is not a pristine external replication or a population-level architecture comparison. The principal application and its original freezes remain unchanged.

## Design

Three new predetermined initialisation seeds, 117, 129 and 143, use the same 200-epoch NumPy Adam recipe and 1,074/238/121/364 train/discovery-fit/discovery-validation/test split. The first hidden layer has 32 units and is the nominated cut. Four fixed eight-unit groups are attenuated at strengths 1/2 and 1 during discovery; test uses those strengths on new images and the unseen strengths 1/4 and 3/4. No label, input identity, intervention identity or full hidden state is supplied to the report decoder.

Candidate reports are native median predicates (8 or 12 state bits), PCA median predicates (8 bits), and a capacity-matched random-direction baseline (8 bits). Each family has 32 candidate linear projections. Random directions are fixed Gaussian unit vectors seeded by model seed plus 5671; PCA is learned from training states. All projection coefficients are quantised at scale 4096 and all thresholds are training order-statistic medians. The same greedy discovery-fit response-MSE procedure selects features, and the same separate discovery-validation ridge choice fits a shared additive response map B[T(h′)−T(h)]. No extension choices are tuned on its test outcomes.

The random direction control has the same dense encoder coefficient budget as PCA. Native8 uses 8 nonzero encoder weights and 80 response coefficients; native12 uses 12 and 120. Dense8 uses at most 256 nonzero encoder weights and 80 response coefficients. Transmission costs are twice the state-bit counts because both endpoint reports are supplied. This does not equalise semantic legibility, which is not measured.

The model weights are integer multiples of 1/4096. Exact endpoint evaluation passes the intervened hidden numerators through the second ReLU using Python arbitrary-precision integers. Centered-logit response differences have denominator 43980465111040. Only after checking the range are target numerators stored as int64. The freeze SHA-256 is 60ee53497911c2b304dd93a3a6db64a35891f1ccf8af80d82a85d46435c03af0. New-model test evaluation began at 2026-10-15T21:29:05Z.

## Held-out results

| Seed | Test accuracy | Export/float class agreement | Largest centered-logit export error |
|---|---:|---:|---:|
|117|98.08%|100.00%|0.006676|
|129|98.63%|100.00%|0.008084|
|143|97.53%|100.00%|0.007782|

All models reached 100% training accuracy. MSE below sums squared error across all ten centered logits and then averages intervention rows. Each bank has 2,912 rows per model; rows sharing an input are dependent.

| Seed | Native8 unseen MSE | Native12 unseen MSE | PCA8 unseen MSE | Random8 unseen MSE |
|---|---:|---:|---:|---:|
|117|32.482|29.584|42.259|38.246|
|129|24.844|21.917|33.828|30.242|
|143|37.975|32.131|51.091|48.377|

The tested additive decoder is more accurate with native predicates than either dense eight-bit baseline in these three models, and PCA is worse than the matched random directions here. This cannot establish general native-feature superiority: it is conditional on the selected one-bit reports, discovery objective, restricted shared decoder and small digit models. All low-bit methods fail the primary maximum-error tolerance of one logit unit.

The exact report-pair conditional-variance floor remains below actual decoder MSE, while report-pair collisions independently falsify information sufficiency at the stated tolerance. Exact affine-circuit certificates on injective report banks, where available, are reported in the separate dual-certificate supplement. The fitted-risk gap alone is not taken as a proof of affine approximation impossibility.

## The nonlinear layer is active in these interventions

A descriptive post-test diagnostic counts second-layer ReLU gates whose sign changes under an intervention. It was not used for selection.

| Seed | Seen strengths: any gate changes | Unseen strengths: any gate changes | Mean changed gates, unseen |
|---|---:|---:|---:|
|117|75.52%|56.28%|0.985|
|129|64.49%|49.59%|0.816|
|143|72.97%|53.74%|0.823|

Thus this is not merely a second layer that stays in one global linear regime. The extension remains a small feedforward model; it does not test attention, recurrence, natural-language semantics or foundation-model prevalence.

## Conservative nonlinear precision certificate

For each first-hidden coordinate, the same exact input-box bound M_j applies. Let C_mk = 10W3_int[m,k]−Σ_lW3_int[m,l]. Since ReLU is coordinatewise 1-Lipschitz, midpoint quantisation with b_j bits gives a paired response bound

Σ_j M_j Σ_m |W2_int[j,m]| |C_mk| /(65536·4096²·10·2^{b_j})

in output k. This is a conservative absolute-weight product bound, not a sharp attainable error or an optimal bit allocation. The greedy allocator and its tie rules are identical in spirit to the primary precision supplement but operate on these fixed larger coefficients. Selection uses model weights only. The decoder evaluates the original nonlinear downstream network on midpoint states, charging its 698 stored weights and biases. This is a numerical compression baseline rather than a simpler semantic algorithm.

| Seed | Uniform state bits | Gain-aware state bits | Certified maximum error | Observed unseen maximum error | Lossless integer box bits |
|---|---:|---:|---:|---:|---:|
|117|320|287|0.994562|0.065583|611|
|129|320|281|0.992009|0.067977|611|
|143|320|286|0.982231|0.051286|610|

Every finite tested precision prediction satisfies epsilon1 by exact integer comparison. The analytic bound covers the whole declared input box and arbitrary coordinate attenuation, including downstream ReLU boundary crossings. Its conservatism is visible in the much smaller observed errors. The lossless code comparison is a coordinatewise fixed-length integer-box benchmark; it is not an optimum joint code or a finite lossless code for a continuous domain.

## Reproduction boundary

Independent arithmetic replay by the same team passed 104 checks and rebuilt all 17,472 target rows and precision predictions using Python integers. The retained code is nonlinear.py; verification/reporting is verify_and_report.py. The sibling application/experiment.py supplies frozen common numerical utilities, with its hash in this supplement’s freeze. No external download or scipy is required for training/evaluation. Keep the two application directories together when reproducing. All model/report matrices, exact targets, original and additional precision parameters, contracts and results are retained.

Dataset attribution and licence are the same as the primary application: Alpaydin and Kaynak (1998), UCI Optical Recognition of Handwritten Digits, DOI https://doi.org/10.24432/C50P49, CC BY 4.0; this uses the preserved 1,797-image scikit-learn subset.
