---
title: "Supplement to Assignment-Aligned Effective States in a 2-SAT Transformer"
author: "Anonymous"
date: "18 October 2026 · Version 4.8.0 · Unrefereed candidate"
lang: en-GB
---

**Version record:** 4.8.0. DOI: [10.5281/zenodo.23454637](https://doi.org/10.5281/zenodo.23454637).

This supplement retains supporting calibrations and the research history. The evidence statuses are those stated for each study. These heterogeneous studies are not pooled into a general claim about native-neuron prevalence, developmental origin or frontier-model safety.

# Appendix A: exact decoder certificates and repair

**Functional compression.** Basu, Seo, and Varshney (2022) define tolerance-compatible sets through common enclosing balls, including decoder side information. Their partially known-function result also transfers distortion guarantees under uniform approximation error. In the maximum norm, the common-ball condition reduces to coordinate ranges, closely matching the earlier paper's graph construction. Our finite worst-case objective differs from their coding-rate objective; that difference alone is not the main advance. The change here is to fix an admissible report grammar and decoder, then expose failures that unrestricted lookup decoding hides.

**Approximation and repair.** The signed nullspace dual and sparse support bound below are classical finite Chebyshev approximation. Yang, Zhang, and Zhang (2023, §3) explicitly discuss both the dual programme and basic-solution sparsity. We use a general finite-matrix argument, without imposing a Haar-system or ordered-alternation assumption. Minimum-cost feature selection, dual decomposition, and counterexample-guided refinement also have substantial precedents: Bertsimas and Van Parys (2020) use exact dual cuts for sparse regression; Aytug (2015) uses decomposition for SVM feature selection; Elboher, Gottschlich, and Katz (2020) refine neural abstractions for verification. Our contrast-cover formulation specifies the corresponding explanation-repair problem. Its novelty cannot rest on relabelling LP duality or refinement.

**Precision allocation.** Hilaire, Ouzia, and Lopez (2019, §VI) minimise total integer word length under worst-case output constraints of the form $\sum_j E_{ij}2^{-w_j+\delta_j}\le\epsilon_i$. This is a close antecedent of our precision optimisation. Their setting concerns fixed-point filters and controllers, including propagated roundoff; ours quantises a single activation cut, uses a known downstream map, and bounds a difference between two outputs. Both charge integer word lengths and use sufficient error bounds. Our additions are the particular neural audit, retained exact rational evidence, and explicit comparison against target-dependent encoders; the optimisation family and Lagrangian principles are established. Lin, Talathi, and Annapureddy (2016) provide related neural bit-allocation work based on signal-to-quantisation-noise modelling, a different guarantee from the worst-case box bounds used here.

| Question | Existing foundation | Role of this paper |
|:--|:--|:--|
| Can an unrestricted decoder represent approximate responses? | Functional compression | Background information-loss oracle |
| Does a proposed intervention map implement a causal abstraction? | Causal abstraction and CAE | Separate mechanism-validity requirement |
| Can this fixed affine response model meet the tolerance? | Chebyshev approximation | Inspectable neural-response failure certificate |
| Which allowed columns can remove every obstruction? | Feature selection and decomposition | Exact menu-relative repair constraints |
| Does a chosen repair transfer, and at what cost? | Controlled evaluation | Frozen trained-system experiment |
| Can native precision guarantee the tolerance? | Quantisation/error propagation | Executed whole-domain constructive comparator |

The search covered these foundations, test cover, discernibility matrices, neural verification, identifiability, and contemporary feature evaluation. The retained search ledger identifies inspected sections and access limitations. No direct implementation of the complete contrast-certificate and repair workflow in mechanistic interpretability was found in that search. This is bounded negative evidence, not a priority proof.

## Affine feasibility and sparse witnesses

For a scalar response $y\in\mathbb R^n$ and fixed matrix $A\in\mathbb R^{n\times d}$, define

$$ e(A,y)=\min_\beta\|y-A\beta\|_\infty. $$

**Theorem 1 (classical minimax dual, used as an audit).**

$$
e(A,y)=\max\{u^\top y:A^\top u=0,\ \|u\|_1\le1\}.
\tag{1}
$$

If $r=\operatorname{rank}(A)$ and $e(A,y)>0$, an optimal witness exists with $\|u\|_1=1$ and at most $r+1$ nonzero entries. A feasible witness with $u^\top y>\varepsilon$ rejects every decoder in this affine class on at least one supported case.

*Proof.* For every feasible $u$,
$u^\top y=u^\top(y-A\beta)\le\|u\|_1\|y-A\beta\|_\infty$.
The primal LP minimises $t$ subject to $y-A\beta\le t\mathbf1$ and $A\beta-y\le t\mathbf1$. Its nonnegative dual multipliers $p,q$ satisfy $A^\top(p-q)=0$ and $\mathbf1^\top(p+q)=1$. Setting $u=p-q$ yields the right side of (1); conversely any such $u$ can be represented with total multiplier mass one by adding cancelling mass. Strong duality proves equality. Choose an extreme point of the nonempty optimal face. A positive optimum has norm one. If its support exceeds $r+1$, a nonzero perturbation $v$ supported there solves both $A^\top v=0$ and $\operatorname{sign}(u)^\top v=0$. Optimality forces $v^\top y=0$: otherwise one of the two small feasible perturbations would increase the objective. Both perturbations therefore preserve signs, norm, feasibility and the optimal objective, contradicting extremality within the optimal face. $\square$

When an intercept is present, positive and negative witness masses are each one half. The witness compares two probability mixtures with identical mean values for every allowed decoder feature but different mean responses. An ordinary report collision is the two-case special case. Sparse does not mean minimum-cardinality; (1) gives a support bound, not an optimisation over witness length.

For independently parameterised output coordinates and maximum error across coordinates, the optimum is $\max_k e(A,y_k)$. A single coordinate rejects the full model. Tied multi-output coefficients require a joint design and do not automatically admit this decomposition.

**Known-answer example.** For binary reports $(0,0),(0,1),(1,0),(1,1)$ and XOR responses $(0,1,1,0)$, all reports are distinct. Nevertheless,

$$A=[\mathbf1,x_1,x_2],\qquad u=(-1,1,1,-1)/4$$

gives $A^\top u=0$ and $u^\top y=1/2$. The constant prediction $1/2$ attains the bound. Adding $\phi=x_1x_2$ permits $x_1+x_2-2\phi$ and zero error. No information was added: the new column is computed from the existing report. The repair enlarges the decoder's expressivity. This is the familiar XOR obstruction, used as a control rather than claimed as a discovery.

## A complete feature-repair criterion

Let $A_0$ be the baseline, $\phi_1,\ldots,\phi_p$ a fixed allowed feature library, and $c_j>0$ their costs. For $S\subseteq[p]$, write $A_S=[A_0,\phi_j:j\in S]$. Minimum repair cost is

$$ C_\varepsilon=\min_{S:e(A_S,y)\le\varepsilon}\sum_{j\in S}c_j, $$

with value infinity if the whole library is inadequate.

**Theorem 2 (contrast constraints).** For every $u$ satisfying $A_0^\top u=0$, $\|u\|_1\le1$, and $u^\top y>\varepsilon$, put $D(u)=\{j:u^\top\phi_j\ne0\}$. Then $S$ is feasible if and only if it intersects every $D(u)$.

*Proof.* If $S$ misses a set $D(u)$, its witness annihilates $A_S$, so Theorem 1 rejects it. Conversely, an infeasible $S$ has a dual witness annihilating $A_S$ with value above tolerance. Its separating set misses $S$. $\square$

This characterisation gives an exact constraint-generation algorithm. Solve the minimum-cost binary covering problem for the retained witness constraints. Audit its selected design by LP. A passing candidate is globally optimal within the fixed library because the covering problem is a lower bound for every feasible repair. A failed candidate supplies a violated cut and is excluded. There are at most $2^p$ selections. An empty $D(u)$ certifies that no repair in the library works. Runtime can be exponential; if stopped early, report the cost gap.

The problem is NP-hard. Let rows be graph edges, candidate columns their endpoint incidences, $A_0$ empty, $y=\mathbf1$, and $\varepsilon=1/2$. A feasible selection must cover every edge. Every vertex cover is feasible with coefficient $1/2$ on selected vertices, so minimum feature count equals minimum vertex cover. This reduction also permits optional coefficient bounds of $1/2$.

Neither a nonzero contrast nor one greedy improvement guarantees a complete repair. For example, with an intercept, $y=(0,1,0)$ and candidates $(0,1,1)$ and $(0,0,1)$, either candidate alone leaves minimax error $1/2$, while both fit exactly. We therefore distinguish the exact algorithm from the four-step greedy heuristic executed on the trained-system bank.

## Coefficient budgets and uncertainty

For optional coefficients constrained by $|\gamma_j|\le b_jz_j$, $z_j\in\{0,1\}$, define $e_b(z;y)=\min\|y-A_0\beta-\Phi\gamma\|_\infty$. LP duality gives

$$
e_b(z;y)=\max_{A_0^\top u=0,\|u\|_1\le1}
\left[u^\top y-\sum_jb_jz_j|u^\top\phi_j|\right].
\tag{2}
$$

*Proof.* The baseline coefficients impose the nullspace constraint. Maximising $u^\top\Phi\gamma$ over the coefficient box gives its support function $\sum_jb_jz_j|u^\top\phi_j|$. Dualising the residual inequalities contributes $\|u\|_1\le1$. The primal is feasible and bounded below, so strong duality applies. $\square$

Consequently every feasible repair must supply enough contrast:

$$\sum_jb_j|u^\top\phi_j|z_j\ge u^\top y-\varepsilon.$$

The full family is sufficient as well as necessary. The unbounded affine sparse-support theorem is not automatically a sparse-support theorem for this penalised dual. In the XOR example, bounding the interaction coefficient by $b$ yields exactly $\max(0,1/2-b/4)$; an arbitrarily weak added feature need not permit the required repair.

Suppose possible responses are $\widehat y+E\eta$, with one shared $\eta$ in a nonempty compact convex set $U$. Let $h_U$ be its support function. Convex minimax applied to (2) yields the sharp smallest attainable restricted error

$$
\inf_{\eta\in U}e_b(z;\widehat y+E\eta)=
\max_{A_0^\top u=0,\|u\|_1\le1}
\left[u^\top\widehat y-h_U(-E^\top u)-\sum_jb_jz_j|u^\top\phi_j|\right].
\tag{3}
$$

The objective is linear in $\eta$ and concave in $u$, and both domains are compact convex; minimising the linear uncertainty term gives $-h_U(-E^\top u)$. A value above tolerance rejects every response bank consistent with the uncertainty model. Correlated uncertainty must remain shared across cases.

For the independent uniform box $\|y-\widehat y\|_\infty\le\delta$ and a nonempty closed prediction set $M$, two different questions have different answers:

$$
\inf_y\operatorname{dist}_\infty(y,M)=\max\{\operatorname{dist}_\infty(\widehat y,M)-\delta,0\},
\tag{4}
$$

$$
\inf_{m\in M}\sup_y\|y-m\|_\infty=\operatorname{dist}_\infty(\widehat y,M)+\delta.
\tag{5}
$$

The reverse triangle inequality proves the lower bound in (4); moving each observed coordinate toward a closest prediction by at most $\delta$ attains it. For fixed $m$, the inner supremum in (5) is $\max_i(|\widehat y_i-m_i|+\delta)$. Equation (4) allows the decoder to adapt to the true response. Equation (5) requires one frozen decoder to handle all possible responses. Even a full-row-rank design has robust error $\delta$ in (5).

## What an exact certificate verifies

The optimiser proposes a support. An independent checker reconstructs rational weights, checks $A^\top u=0$ and $\|u\|_1=1$ exactly, and computes $u^\top y$ from integer response numerators. A dyadic exported primal decoder supplies a separately computed exact upper bound. Tight lower/upper brackets do not require trust in a floating-point solver's success flag. They certify only the retained finite instance.

Approximate annihilation is insufficient with unbounded coefficients. If $A'=A+\Delta A$, $y'=y+\Delta y$, and $\|\beta\|_1\le B$, a normalised witness transports to the lower bound

$$u^\top y-\|\Delta y\|_\infty-B\|\Delta A^\top u\|_\infty.$$

This follows by expanding $u^\top(y'-A'\beta)$ and applying Hölder's inequality. Without $B$, an arbitrarily small missing feature component can be amplified arbitrarily. This matched-case bound does not certify new interventions or population coverage.

# Appendix B: trained-digit calibration and complete results

## Study protocol

### Dataset, model, and chronology

We use the 1,797-image, $8\times8$ optical-digit dataset bundled with scikit-learn, derived from the UCI optical-recognition dataset. The stratified split is fixed at 1,074 training images, 238 discovery-fit images, 121 discovery-validation images, and 364 test images. Writer identities are unavailable in the retained file, so this is an image holdout, not a writer-independent evaluation. The split seed is 20261015; network seeds are 17, 29, and 43.

Each classifier has architecture $64\to32\to10$, ReLU hidden activations, and a linear output head. Training inputs are integer pixels $x_j\in\{0,\ldots,16\}$ divided by 16. NumPy training uses 200 fixed Adam epochs, batch size 128, learning rate 0.003 for 150 epochs and 0.001 thereafter, and weight decay $10^{-4}$. No test-based early stopping is used. We round every weight and bias to an integer divided by 4096. The hidden integer numerator is $h^{\rm int}=\max(xW_1^{\rm int}+16b_1^{\rm int},0)$, with physical activation $h=h^{\rm int}/65536$. Quarter-strength interventions use numerators on the further denominator-four grid. Centred logit changes then have retained integer numerators with denominator 10,737,418,240. These are the nominated responses; they are not claims about exact softmax probabilities.

The initial contract and data split precede training and feature fitting. Model, encoder, and decoder files are hashed before test evaluation. The contrast-selected twelve-predicate decoder and precision supplements were designed after discovery results but also frozen before test outcomes were opened. They are supplementary, not part of the initial contract or an external preregistration. The contrast supplement's eight-predicate minimax comparator was recomputed later using discovery data only, and its test-evaluation script was written after test opening; these do not inherit the twelve-predicate freeze. The full-library impossibility analysis is retrospective deterministic analysis of discovery data. All selected seeds and failed comparisons are retained.

### Interventions and reports

Four disjoint eight-neuron groups are selected by a fixed permutation. An intervention multiplies the nominated group's activations by $1-s$ and leaves the other hidden units fixed. Discovery uses $s=1/2$ and $1$. Test banks use both those strengths and the unseen strengths $1/4$ and $3/4$ on held-out images. Each strength bank has 2,912 rows per model. A separate bank of 364 cyclic pairs of test images replaces the entire hidden state by another natural hidden state. Such replacements are not the same distribution as group attenuation.

The attenuation intervention is mathematically well-defined at the nominated hidden cut. We do not assert that every attenuated state corresponds to a natural image. No intervention identifier, image identity, label, or uncharged hidden state is supplied to the decoder.

Native reports threshold individual neurons. Group reports threshold sums of paired neurons, with pairs chosen by greedy training-set coactivation correlation. The original contract described these correlations as positive, but the last available pair need not be positive; a deviation note preserves that wording and records the actual procedure. PCA reports threshold learned training-covariance directions with coefficients rounded to the same integer scale. These are principal-component baselines, not sparse autoencoders or semantically validated features.

All families use 4, 8, 12, or 16 binary state predicates. The same discovery-fit response-MSE ranking procedure selects predicates for each family. Thresholds are training medians. The initial shared response decoder is

$$\widehat Y=B\,[T(H')-T(H)],$$

with no intercept, interaction terms, or action-specific coefficients. Ridge regularisation is chosen from five prespecified values on the separate discovery-validation split; the final decoder is fitted on discovery-fit only. Encoder supervision uses discovery responses and is disclosed. These are constrained response models, not independently written human descriptions.

The original repair adds four native quartile predicates to the eight-bit native report, selecting by discovery-fit MSE. The equal-bit comparator is the twelve-predicate native median report produced by the original ranking. The contrast supplement permits an intercept, making its class more permissive. At each of four additions, it audits all ten discovery response coordinates, takes the largest minimax error, and selects the remaining predicate with largest absolute exact witness contrast. This is a heuristic; the exact contrast-cover algorithm is separately validated on small instances.

### Costs and primary decision

The primary tolerance is one centred-logit unit in every response coordinate. This is an operational study threshold, not a safety threshold. Tolerance sensitivity uses 0.25, 0.5, 1, 2, and 4. A $B$-predicate report transmits $B$ bits per state and $2B$ per pair. Its shared decoder uses $10B$ binary64 coefficients. Native predicates use one encoder weight each, paired sums use two, and PCA predicates can use up to 32. Thus equal bit counts do not mean equal total computational or description cost.

We distinguish mean squared error, maximum error, and exact within-report incompatibility. The response bank is deterministic. Exact arithmetic supplies its numerical certificate; no confidence interval is needed for its finite maximum. The three initialisations are not three independent datasets, and interventions of the same image are dependent. We report each seed rather than treating thousands of intervention rows as independent scientific replications.

## Results: four different reasons an explanation can fail

### Distinct reports do not establish affine adequacy

The exported classifiers attain test accuracies 98.90%, 98.63%, and 98.08%. For each model, retain the first discovery row for each distinct eight-predicate difference vector and add an intercept. Every report on this subbank is distinct, so an unrestricted lookup decoder has exactly zero empirical error. Nevertheless, exact affine witnesses reject the primary tolerance:

| Seed | Witness cases | Exact lower bound | Approximate value |
|---:|---:|:--|---:|
| 17 | 4 | $13183958133/10737418240$ | 1.2279 |
| 29 | 4 | $8324109771/4294967296$ | 1.9381 |
| 43 | 6 | $26016914143/16106127360$ | 1.6153 |

These are trained-network response examples, not an inserted XOR circuit. Deduplication deliberately constructs a diagnostic subbank rather than an estimate of prevalence. It preserves actual model responses and establishes existence of an expressivity failure beyond pairwise report collisions. The finite subbank has no held-out generalisation claim. Each certificate remains a strict rejection under an additional independent uniform response allowance smaller than its lower bound minus one.

### Average-error repair is partial and does not dominate equal cost

The original four-quartile-predicate repair improves mean squared error on unseen intervention strengths relative to eight native predicates in all three models. It does not consistently improve over twelve native median predicates:

| Seed | Native 8 bits | Native 12 bits | Quartile repair 12 bits |
|---:|---:|---:|---:|
| 17 | 17.071 | 15.208 | 14.849 |
| 29 | 16.458 | 14.111 | 14.581 |
| 43 | 16.775 | 13.880 | 13.991 |

Values are mean squared response error summed over ten outputs. The primary worst-case tolerance still fails. This comparison does not support a general claim that the proposed greedy refinement is superior. It supports a narrower diagnostic statement: additional precision in selected native predicates can improve average response prediction, but that improvement is insufficient evidence of adequacy or best use of a fixed budget.

| Report | Seen-strength MSE | Unseen-strength MSE | Natural-pair MSE |
|:--|---:|---:|---:|
| Native 4 | 37.286 | 21.572 | 645.492 |
| Native 8 | 28.375 | 16.768 | 510.724 |
| Native 12 | 23.170 | 14.400 | 408.756 |
| Quartile repair 12 | 22.143 | 14.473 | 411.776 |
| Native 16 | 19.400 | 12.824 | 330.559 |
| Group 8 | 30.728 | 18.639 | 535.890 |
| Group 16 | 26.188 | 16.491 | 431.925 |
| PCA 8 | 42.358 | 23.061 | 816.225 |
| PCA 16 | 37.100 | 20.923 | 737.382 |

These are descriptive means over the three models; the archive retains every seed and budget. The unseen-strength bank has smaller response magnitudes, so a lower raw MSE there is not by itself stronger transfer. The natural-pair bank tests a different, harder intervention family. All its fitted predictions fail the primary simultaneous-output tolerance. PCA8 has distinct natural report pairs for every case in two models, but its fitted response MSE exceeds 800 in both; finite lookup injectivity is again a weak criterion.

![Held-out response error versus report budget. Lines show means over three model seeds; shaded regions show the minimum–maximum range, not confidence intervals. Every seed is retained in the data tables. Bit count does not charge the larger PCA encoder. The repair methods are compared in tables and are not plotted here.](figures/budget_transfer.pdf){width=95%}

The contrast heuristic lowers the worst discovery-coordinate minimax error from 8.2135 to 6.3390, 6.7024 to 5.9212, and 9.9300 to 6.6040. On unseen strengths, its fixed minimax decoder has maximum errors 8.3834, 7.8349, and 8.5218, compared with 10.4075, 8.2440, and 8.3717 for the corresponding eight-predicate affine baseline. The third model becomes slightly worse on that held-out maximum. Its mean squared error also worsens for seed 29. The theorem certifies discovery feasibility; it does not make greedy selection optimal or protect against distribution shift.

A post-test exploratory comparison equalises the decoder objective and budget. Native12 and contrast12 are both fitted with an intercept, ten independent minimax outputs, and coefficients rounded to denominator $2^{32}$. A third method starts from native8 and exhaustively evaluates each remaining quartile candidate at each of four greedy additions. It selects the candidate giving the smallest discovery maximum over outputs; this is a numerical greedy rule, not globally optimal subset search. Identical design rows are reduced to their response extrema during candidate search, preserving the LP inequalities exactly. Final fits and certificates use the full rows.

| Seed | Native12 unseen maximum | Contrast12 | Direct greedy minimax12 |
|---:|---:|---:|---:|
| 17 | 8.0728 | 8.3834 | 7.6478 |
| 29 | 8.0044 | 7.8349 | 7.8349 |
| 43 | 7.7431 | 8.5218 | 8.6144 |

The full candidate search solves 9,660 scalar LPs in approximately 9.1 seconds on the recorded local environment. Contrast guidance does not consistently outperform either matched comparator. It has lower unseen MSE than direct greedy minimax in all three seeds, but lower MSE than native12 only in seed 43; improvement depends on the objective. Every method fails tolerance one. The 120 discovery output brackets and 36 test maxima have exact replay. Selection used discovery responses, but the design followed inspection of the earlier results and reuses the opened test bank. It closes a descriptive control gap without establishing new protected generalisation.

### The existing predicate library cannot supply the requested repair

A retrospective discovery audit gives every native quartile candidate to the affine decoder, together with an intercept. Even this more permissive class fails:

| Seed | Available predicates | Certified lower error | Witness cases |
|---:|---:|---:|---:|
| 17 | 89 | 2.5704 | 25 |
| 29 | 94 | 2.7133 | 19 |
| 43 | 87 | 3.2497 | 25 |

The rational lower bounds and separately evaluated dyadic upper bounds agree to less than $10^{-7}$. Because the witnesses annihilate the entire library, every subset fails the primary tolerance. Searching more combinations of the same predicates cannot fix that conclusion. The appropriate next action is to enlarge the available precision or decoder class, change the target/tolerance, or report failure. It is not to interpret another fitted loss reduction as proof of sufficiency.

A separate post-test oracle sensitivity check supplies the exact intervention group and strength as context. The eight-bit native report still has minimum worst-case unseen-strength errors 3.8671, 3.9778, and 4.4623. The retained witnesses therefore share an action and magnitude: failure is not solely caused by withholding action identity. This is a descriptive finite-bank calculation, not a newly fitted predictor.

![A held-out same-intervention information-loss witness. Distinct source images receive identical native report pairs under the same group attenuation, yet the actual response differs beyond tolerance. This complements the multi-case affine witnesses, which have distinct reports.](figures/collision_witness.pdf){width=92%}

### Native precision supplies a certified repair at a higher cost

For the exact exported model, the downstream map is linear in $h$. Let $c_{jk}$ be its centred readout coefficients and let $0\le h_j\le M_j$ be a certified activation box. Quantise each coordinate into $2^{b_j}$ equal intervals and decode by the interval midpoint. Then each coordinate error is at most $M_j/2^{b_j+1}$. For any pair of states in the box,

$$
\max_k|\Delta F_k-\Delta\widetilde F_k|
\le\max_k\sum_j |c_{jk}|M_j2^{-b_j}.
\tag{6}
$$

This follows by applying the triangle inequality to the two quantisation errors. The right side is sharp as a supremum over independent activation-box pairs with a fixed readout, but the activation box can overapproximate states reachable from real inputs. For our model, $M_j$ follows directly from the positive parts of first-layer weights over the entire pixel box $[0,16]^{64}$. Group attenuation preserves the box. The bound therefore covers continuous inputs in that box and arbitrary attenuation strengths in $[0,1]$, including the held-out strengths.

Uniform eight-bit precision per neuron uses 256 bits per state and certifies tolerance one in every model. A deterministic gain-aware allocation, frozen before test evaluation, uses:

| Seed | Bits per state | Certified bound | Uniform bits |
|---:|---:|---:|---:|
| 17 | 220 | 0.985845 | 256 |
| 29 | 230 | 0.993470 | 256 |
| 43 | 212 | 0.982997 | 256 |

All retained test responses satisfy their exact certified inequalities. This is a successful tolerance repair, although it has a much larger transmission cost than the binary-predicate models and retains the network's known readout. It does not provide semantic labels or discover a new causal mechanism. The exact coordinatewise integer-state encoding costs 625, 629, and 616 bits before quarter-strength scaling; it is a fairer finite-precision reference than an arbitrary floating-point storage width. Continuous-state exact lossless coding has no finite-bit analogue.

The experiment consequently changes the audit decision in a concrete way: the tested low-bit reports are rejected; a small average-error repair is labelled insufficient; the full quartile menu is proved inadequate under affine decoding; and an explicitly costed precision repair is accepted on a larger declared domain.

### Exact cost of the box-certified precision repair

A further, post-test weight-only optimisation asks for the smallest total integer bit count satisfying (6). It does not read test responses or retest a selected predictor. Let $a_{jk}=|c_{jk}|M_j$. The optimisation is

$$
\min_{b_j\in\mathbb Z_{\ge0}}\sum_j b_j
\quad\text{subject to}\quad
\sum_j a_{jk}2^{-b_j}\le\varepsilon\quad\text{for every }k.
\tag{7}
$$

For any nonnegative rational multiplier vector $\mu$, every feasible allocation has cost at least

$$
L(\mu)=\sum_j\min_{b\in\mathbb Z_{\ge0}}
\left[b+2^{-b}\sum_k\mu_k a_{jk}\right]
-\varepsilon\sum_k\mu_k.
\tag{8}
$$

*Proof.* Add $\sum_k\mu_k(\sum_ja_{jk}2^{-b_j}-\varepsilon)\le0$ to a feasible objective and minimise the resulting separable expression over all nonnegative integer bits. If $s=\sum_k\mu_ka_{jk}$, the difference between successive values of $b+s2^{-b}$ is $1-s2^{-b-1}$. Its sign locates a global integer minimum exactly. $\square$

Thus $\lceil L(\mu)\rceil$ is an integer lower certificate, not a solver-status claim. Exact feasible allocations and rational multipliers establish costs 218 and 211 for seeds 17 and 43: their lower bounds are respectively 217.5161565 and 210.2684407. For seed 29, the rational lower bound 228.733 leaves a one-bit gap below the feasible 230-bit allocation. Exact reduced-cost restrictions and a finite integer search exclude every allocation of cost at most 229 in 581 search nodes. The search uses exact per-output suffix-knapsack lower bounds; its full algorithm and retained inputs accompany the proof.

The complete tolerance sweep gives the following certified optima for (7):

| Tolerance | Seed 17 | Seed 29 | Seed 43 |
|---:|---:|---:|---:|
| 0.25 | 282 | 294 | 271 |
| 0.5 | 250 | 262 | 241 |
| 1 | 218 | 230 | 211 |
| 2 | 187 | 198 | 181 |
| 4 | 156 | 166 | 151 |

These optima concern coordinatewise midpoint quantisation in the native basis with the fixed readout and stated box guarantee. They are not minimum costs over all bases or encoders, semantic explanations, or reachable-state-specific guarantees. The practical increment over the greedy allocation is modest; the methodological difference is that the remaining cost uncertainty is closed rather than inferred from a heuristic or a floating-point mixed-integer gap.

### A target-dependent code is much shorter

A retrospective weight-only control moves the known readout into the encoder. Let $p_k(h)=c_k^\top h$. Its box range has endpoints $\ell_k=\sum_jM_j\min(c_{jk},0)$ and $u_k=\sum_jM_j\max(c_{jk},0)$, and width $w_k=u_k-\ell_k$. Separately quantise each projection into $2^{b_k}$ midpoint bins. Choosing the smallest nonnegative integer $b_k$ with $w_k2^{-b_k}\le\varepsilon$ gives a paired-response guarantee by the same two-endpoint triangle inequality. The paired decoder simply multiplies each bin-index difference by its bin width. Offset and half-bin terms cancel.

For tolerance one this requires **77 bits per state in all three models**, or 154 bits per pair, with exact maximum certificates 0.988321129, 0.980876435, and 0.961297677. Bit choices and the failure of each one-bit-smaller choice are checked with fractions. This is an optimum for these ten independent width constraints, not for all response codes; the construction does not exploit the sum-zero dependence between centred outputs. No test outcomes were used for this control.

The contrast with 218, 230, and 211 neuron-local bits is substantial, but it does not discover a simpler neural mechanism. The encoder now computes the target projections themselves. It must therefore be charged for all 320 readout coefficients and their arithmetic. The comparison demonstrates why decoder simplicity alone is an inadequate explanatory budget and why an admissible encoder grammar must be specified.

| Report at tolerance one | State bits | Pair bits | Encoder specification | Paired decoder | Selection information |
|:--|---:|---:|:--|:--|:--|
| Native 8 predicates (fails) | 8 | 16 | 8 unit indices and thresholds | 80 binary64 coefficients | Discovery responses |
| PCA 8 predicates (fails) | 8 | 16 | Up to 256 integer direction weights; 8 thresholds | 80 binary64 coefficients | Training covariance; discovery responses |
| Native precision, frozen | 220 / 230 / 212 | 440 / 460 / 424 | 32 ranges and integer bit depths | 320 known centred-head coefficients | Model weights only |
| Native precision, optimised | 218 / 230 / 211 | 436 / 460 / 422 | Same grammar | Same known head | Post-test, weights only |
| Output projections | 77 / 77 / 77 | 154 / 154 / 154 | 320 known head coefficients; 10 offsets, widths and bit depths | 10 diagonal bin-width scales | Post-test, weights only |

This ledger separates transmitted bits from stored specifications and computation. Encoder integers derive from weights rounded at scale 4096; centred-head coefficients have denominator $10\cdot4096$. Bin widths and decoder scales are derived rational constants, so shared constants need be stored only once. The table is not a single universal description length and does not measure human comprehension. Both precision constructions are valid response codes under their stated contracts; neither alone validates an intervention-transition explanation.

# Appendix C: nonlinear downstream calibration

A second architecture checks whether the numerical failures require a final linear-head cut. Three new models, seeds 117, 129, and 143, have architecture $64\to32\to16\to10$ with ReLU at both hidden layers. We intervene at the first hidden layer, so the downstream computation includes another ReLU. Training and the image split follow the preceding recipe. The extension was designed after the shallow test outcomes had been inspected and reuses those images. Its choices were fixed before evaluating the new models, but it is exploratory rather than a pristine external replication.

Native median predicates, PCA directions, and Gaussian random directions each have a 32-candidate pool. The dense directions use the same integer coefficient precision; all families use the same discovery-selection and ridge-validation procedure. Native reports use 8 or 12 bits; PCA and random reports use 8. The random-direction comparator matches PCA's encoder size. This controls a specific direction-learning advantage; it does not compare semantic interpretability.

| Seed | Accuracy | Native 8 unseen MSE | Native 12 | PCA 8 | Random 8 | Unseen interventions changing a downstream ReLU gate |
|---:|---:|---:|---:|---:|---:|---:|
| 117 | 98.08% | 32.482 | 29.584 | 42.259 | 38.246 | 56.28% |
| 129 | 98.63% | 24.844 | 21.917 | 33.828 | 30.242 | 49.59% |
| 143 | 97.53% | 37.975 | 32.131 | 51.091 | 48.377 | 53.74% |

All three reach 100% training accuracy, and the integer exports preserve every tested class prediction of their floating-point models. Each unseen bank contains 2,912 intervention rows. Errors again sum squared discrepancies over ten centred outputs before averaging. Native predicates outperform the tested dense eight-bit reports here; PCA also performs worse than matched random directions. These descriptive comparisons support no general ranking of representation families. Every low-bit model fails the primary maximum-error tolerance. The gate-change diagnostic establishes that the downstream network is not remaining in one global linear regime.

The injective-subbank audit also survives the nonlinear cut. In each model, retaining one row per distinct native8 signed report gives 15 discovery cases and zero unrestricted lookup error. Four-case rational contrasts certify affine errors at least 1.932362, 2.866955, and 1.954109 for seeds 117, 129, and 143. Exact dyadic primal bounds match these lower bounds to numerical-export precision. These are deterministic existence certificates on discovery cases; they do not estimate the frequency of an interpretability failure.

The same precision approach supplies a conservative repair. If $V$ is the second-layer matrix and $c$ the centred final head, ReLU's coordinatewise 1-Lipschitz property gives the paired bound

$$\max_k\sum_j M_j\sum_m|V_{jm}||c_{mk}|\,2^{-b_j}.$$

Gain-aware allocations use 287, 281, and 286 state bits, compared with 320 uniform bits, and certify maximum errors 0.994562, 0.992009, and 0.982231 on the whole declared box. Their observed unseen maxima are 0.065583, 0.067977, and 0.051286; the large gap illustrates conservatism. The decoder runs the original downstream nonlinear network and is charged all 698 weights and biases. These are sufficient bounds, not optimal allocations or semantic mechanisms. Independent integer replay reconstructs all 17,472 target rows and checks the retained precision predictions. The extension removes one narrow architectural limitation while leaving the main significance question unresolved.

# Appendix D: prospective state and action calibrations

The projection control exposes a limitation of the paired-response contract. If an encoder can compute the output at both endpoints, the response can be coded accurately without describing how an intervention changes the state. We therefore add a separate, exploratory prospective audit. Its decoder receives the original report and the specified action, with no true post-action state or report. This distinction belongs to established causal-abstraction, compositional-validation and predictive-state theory. Palumbo et al. (2025) formalise component and prefix equivalence and replaceability; Vafa et al. (2024) show that successful output prediction can coexist with incoherent world models. We do not claim to introduce the general need for compositional validation.

## Exact action dependence of a linear state report

Write the centred output as $Ch+c_0$, with $C\in\mathbb R^{10\times32}$ and fixed known bias $c_0$, and the allowed state transformations as $h\mapsto D_a h$. A linear report $Rh$ admits an exact affine output decoder $O(Rh)+c_0$ and linear report-transition maps precisely when

$$C=OR,\qquad RD_a=K_aR\quad\text{for every }a.$$

Its minimum possible dimension is the dimension of

$$V=\operatorname{span}\{\operatorname{rows}(CD_w):w\text{ is a finite action word, including the empty word}\}.$$

*Proof.* Any report satisfying the factorisations contains every row of $CD_w$ in its row space, by induction on word length. Conversely, $V$ contains the rows of $C$ and is invariant under right multiplication by every $D_a$. A basis matrix of $V$ therefore admits both factorisations. Equivalently, $\ker R$ must be invisible to $C$ and invariant under every action. $\square$

This is the classical observable quotient for switched linear systems (Petreczky, Bako, and van Schuppen, 2013). Iterating $V\leftarrow V+\sum_aVD_a$ computes it in at most $32-\operatorname{rank}C$ strict dimension increases. It is not a new representation-learning theorem.

For disjoint coordinate-group attenuations $D_g=I-\alpha_gP_g$ with $\alpha_g\ne0$, assume first that the groups partition every state coordinate, so $\sum_gP_g=I$. The closure is then $\bigoplus_g\operatorname{row}(CP_g)$: each group component is obtained from $(C-CD_g)/\alpha_g$, and products of the disjoint projectors add nothing. If some coordinates are never attenuated, put $P_0=I-\sum_gP_g$ and include the additional summand $\operatorname{row}(CP_0)$. It is obtained by subtracting the attenuated components from $C$ and is unchanged by every action. The sum is direct because its summands have disjoint coordinate supports. The fixed known output bias requires no state coordinate and is added by the output decoder, not suppressed by these row-space identities.

The experiment's four eight-neuron groups partition all 32 hidden coordinates. The three shallow models all have exact current-output rank nine. Coarsening the groups into the two specified 16-neuron supergroups $(G_0\cup G_1,G_2\cup G_3)$ gives ranks nine plus nine and hence action-closed rank eighteen; retaining all four original groups gives rank thirty-two. Each eight-neuron group block has rank eight. Nonzero integer minors and algebraic upper bounds certify these ranks, without a singular-value threshold. A separate set of 98 constructed admissible inputs has hidden states spanning all 32 coordinates, so the necessity for **linear** decoder/transition identities already holds on reachable states. A finite nonlinear lookup decoder is outside that rank claim.

## Identical exact current outputs can require different action predictions

The stronger, decoder-independent construction finds pairs of continuous inputs in $[0,16]^{64}$ with identical complete current centred outputs and different responses to the same group ablation. It uses weights only, without sampled image or test outcomes. At the centre input $x_0=(8,\ldots,8)$, let $A$ be the fixed first-layer ReLU active mask. Within that activation cell, an input perturbation $v$ preserving the current output satisfies

$$C A(W_1^{\rm int})^\top v=0.$$

We constrain $|v_i|\le8$ and

$$\left|[(W_1^{\rm int})^\top v]_j\right|\le\left|8\sum_iW^{\rm int}_{1,ij}+16b^{\rm int}_{1,j}\right|,$$

so $x_0\pm v$ remain in the pixel box and the same closed activation cell. An LP proposes a direction separating one group response. A rational projection onto the exact nullspace and rational rescaling produce the final inputs. Re-evaluating the entire exported model with fractions verifies their admissibility and equality of all current outputs; the proof does not rely on the LP status.

| Seed | Ablated group | Output coordinate | Exact-response gap, rounded | Forced prospective maximum error |
|---:|---:|---:|---:|---:|
| 17 | 3 | 1 | 11.019541 | 5.509771 |
| 29 | 0 | 2 | 10.282776 | 5.141388 |
| 43 | 2 | 4 | 9.179579 | 4.589789 |

The last column is half the gap. Any decoder receives the same report and action for both inputs and must return the same prediction, so the triangle inequality forces that error on at least one endpoint. The result applies to every report $T(h)=f(Ch+c_0)$, even one retaining the exact current output with unlimited precision. It is not a quantisation or fitting failure. The pairs are constructed continuous inputs; they are not natural digit images or a population failure-rate estimate.

Approximate prospective sufficiency need not preserve a full exact state. If $\widetilde h$ is the midpoint reconstruction of an initial native code and $D$ is any composition of the permitted diagonal attenuations, so $0\le D_{jj}\le1$, then both $CD\widetilde h+c_0$ and $C(D-I)\widetilde h$ have errors bounded by

$$\frac12\max_k\sum_j|C_{kj}|M_j2^{-b_j}.$$

The factors $|D_{jj}|$ and $|D_{jj}-1|$ are at most one, which proves the bound. The previously certified allocations at paired tolerance two therefore give prospective tolerance-one bounds at 187, 198, and 181 bits per initial state. These are restricted sufficient-bound optima, not universal prospective code minima. The decoder is given the exact action or action word separately and evolves the approximate initial state without repeatedly quantising it. This does not assert exact transition closure of the finite quantiser code.

## A decision failure on natural images

The class-flip policy receives the eight-bit report and ordered top/runner labels, with neither the image nor its exact baseline margin. Conversely, a task-specific encoder could compute all four action values and transmit an optimal action in two bits. That code pays for action evaluation inside the encoder and solves a different task from response or state reconstruction. Its small transmission cost does not contradict the linear closure or precision results.

The exploratory prospective experiment reuses all six models and the already inspected split. It fits an action-conditioned decoder $sB_g[1,z]+s(1-s)G_g[1,z]$, using discovery strengths $1/2$ and $1$ and separate validation. For a $d$-coordinate report this uses $80(d+1)$ coefficients. Queries additionally supply four bits identifying the group and one of four tested strengths. These are more permissive decoders than the earlier shared difference model, so cross-study errors cannot isolate information timing alone.

Current-output codes predict natural-image action effects better than the eight-bit reports in all six models, despite their structural failure above. Their unseen MSEs are 0.682, 0.845, and 0.569 in the shallow models and 3.481, 2.696, and 3.413 in the nonlinear models. Their observed maximum errors still exceed one. Every current-output code is unique among the 364 natural test images in each model; its finite lookup oracle therefore has zero error. The fitted failures cannot be used as information-loss certificates. Native precision controls predict from a reconstructed initial state and the action, meet the bound, and charge the original downstream computation.

For a fixed 75% attenuation, we also ask which of the four groups most decreases the original top-class-versus-runner-up margin. The policy receives only the native8 report and the ordered class labels, costing at most seven additional context bits. Let $v_i(a)$ be the margin decrease and $r_i(a)=\max_bv_i(b)-v_i(a)$ the action regret. On each identical-report/context class $Q$, the smallest worst-case regret of any deterministic policy is exactly

$$\min_a\max_{i\in Q}r_i(a).$$

*Proof.* A deterministic policy must choose one common action on $Q$, so every policy has at least that value. Choosing its minimising action in each class attains it. The whole-bank optimum is the largest class value. The corresponding empirical mean floor minimises the sum within each class. $\square$

| Seed | Exact minimum worst-case regret, rounded | Minimum empirical mean regret |
|---:|---:|---:|
| 17 | 1.7225 | 0.0184 |
| 29 | 2.4091 | 0.0256 |
| 43 | 3.3756 | 0.0452 |
| 117 | 1.8672 | 0.0191 |
| 129 | 3.4236 | 0.0776 |
| 143 | 4.8722 | 0.0774 |

Enumeration of four actions using integer responses verifies every value; a two-image witness attains each worst-case obstruction. The small mean floors matter: the tail examples do not imply that poor action choices dominate average behaviour. A randomised policy under expected regret defines a different optimisation, not the table's claim.

One pair also has a direct classification consequence. In shallow model 43, natural digit-eight images 1305 and 253 share their eight-bit report, predicted class eight, and runner-up one. Their original margins are 5.17655 and 4.53109. After actions zero through three, the first image's margins are $(6.37360,9.04904,-0.86463,2.51096)$ and the second's are $(6.41540,7.18880,2.47000,-1.10297)$. Only action two changes the first prediction from eight to one; only action three does so for the second. Every deterministic policy restricted to their common report and context therefore fails on at least one, although a successful action exists for each. Uniformly over this particular pair its success cannot exceed one half; this is not the test-set failure rate.

![Two natural images with the same native report require different classification-changing actions. At the fixed 75% attenuation, only group two flips image 1305 from eight to one, while only group three flips image 253. The certificate concerns this pair and the stated intervention menu.](figures/prospective-decision-seed43.pdf){width=95%}

The largest margin-regret witness, from nonlinear model 143, illustrates why the target must remain explicit: its two images have different best actions for reducing the one-versus-eight margin, yet the same action can change both full predictions through other classes. Margin regret and arbitrary class flipping are not interchangeable. All fixed examples, complete action tables and both illustrations are retained. Independent integer replay reconstructs 34,944 prospective response rows and every fitted prediction from initial code and action alone.

These results make some report failures consequential for an actual intervention choice. They still concern automatically selected predicates in six small models on one reused dataset. They do not adjudicate an independently motivated semantic explanation of a large model, establish a general repair advantage, or turn the classical observable-quotient argument into a novel theory.

# Appendix E: complete finite exclusion for integer precision

The integer optimum in (7) needs an exclusion argument whenever rounding the Lagrange bound (8) leaves a gap. This appendix supplies that argument without relying on mixed-integer solver status.

For a fixed nonnegative rational multiplier $\mu$, put $s_j=\sum_k\mu_ka_{jk}$, $m_j=\min_{b\ge0,\,b\in\mathbb Z}(b+s_j2^{-b})$, and $\Delta_j(b)=b+s_j2^{-b}-m_j\ge0$. The sign of $1-s_j2^{-b-1}$ locates each global discrete minimum. If an allocation is feasible and has total cost at most an integer $B$, then

$$\sum_j\Delta_j(b_j)\le B-L(\mu).$$

Indeed, its Lagrangian value equals $L(\mu)+\sum_j\Delta_j(b_j)$ and cannot exceed its feasible objective. Since $b+s_j2^{-b}\ge b$, every coordinate must belong to the finite set

$$S_j(B)=\left\{b\in\mathbb Z_{\ge0}: b\le\lfloor B-L(\mu)+m_j\rfloor,\quad\Delta_j(b)\le B-L(\mu)\right\}.$$

These are safe restrictions on an unbounded integer problem, not an assumed maximum bit depth. Branch over every allowed bit depth in each $S_j(B)$. Reject a partial assignment if its accumulated cost exceeds $B$, its reduced penalty exceeds $B-L(\mu)$, or even the smallest possible remaining contribution violates an output constraint.

The last rejection uses a suffix dynamic program. For output $k$, remaining units indexed from $i$, and remaining budget $c$, let

$$Q_{i,k}(c)=\min_{b\in S_i(B),\,b\le c}\left\{a_{ik}2^{-b}+Q_{i+1,k}(c-b)\right\},\qquad Q_{n+1,k}(c)=0\quad(c\ge0).$$

An empty choice set has value infinity. Different outputs may minimise with different assignments; this relaxes the common-assignment problem, making each $Q$ a valid lower bound. If a partial output contribution plus $Q_{i,k}(c)$ exceeds $\varepsilon$ for any $k$, no completion of that branch is feasible. Otherwise visit all its permitted next choices. Termination without a feasible leaf proves that no allocation with cost at most $B$ exists.

At tolerance one for seed 29, the exact Lagrange bound is

$$L(\mu)=\frac{9824022476083989677277}{42949672960000000000}.$$

Set $B=229$. There are 23 binary choices after the safe restrictions; all others are fixed. Clearing denominators gives response denominator 687194767360 and penalty denominator 42949672960000000000. Exact search visits 581 nodes, rejects 113 by reduced penalty and 178 by response/cost bounds, and reaches no feasible leaf. The retained 230-bit allocation is checked directly, proving optimality. Seeds 17 and 43 need no exclusion at this tolerance because their rational lower bounds round to their feasible costs. The same procedure verifies all fifteen tolerance/seed cases in the table. An independently written reviewer search using a different branching rule and no submitted suffix dynamic program agrees on every optimum.

The source, coefficients, bit vectors, multipliers, finite permitted sets, and exact receipts are in the accompanying bundle. The proof above explains why those finite checks establish the stated restricted optima. It does not convert a box-bound optimum into an unrestricted communication lower bound.

# Appendix F: Inception source calibration and normalisation provenance

The canonical `mixed4e:55` case requires model identity before semantic adjudication. We executed the original Lucid TensorFlow GraphDef on the essay's twelve author-selected WebP illustrations: six labelled pointed and six round. Complete framed images were resized bilinearly to $224\times224$, with input transformation $255\,\mathrm{RGB}-117$. Before inference, display-marker locations were mapped to the nearest cell of the graph-derived lattice $15.5+16j$, $j=0,\ldots,13$, with lower-index ties. The target is zero-based channel 55 before ReLU in the first $1\times1$ branch of `mixed4e`. This does not recover historical crops or tensor coordinates; the failed PyTorch parity attempt was not used.

All six pointed responses were positive and all six round responses negative. Fixed bilinear sensitivity readouts agreed in sign, and two separate-process replays had zero discrepancy. Fourteen forwards completed the calibration; its descriptive 95th–5th percentile scale was $S=539.265894$. These selected illustrations are **not independent evidence for geometry**, and contain no verified historical natural cat-leg example. The recovered cat-leg montage is synthetic. Narang's (2026) quantitative channel-decomposition study is relevant prior art, not an experiment replicated here.

A paired within-TensorFlow comparison changed only both local-response-normalisation operators. The native eleven-channel sum uses radius five, bias two and $\alpha=10^{-4}$; the Lucent convention uses a nine-channel average, bias one and the same nominal $\alpha$, represented in TensorFlow by radius four and $\alpha/9$. Both retain exponent 0.5. All weights, inputs, cells and other graph attributes remained identical. Fourteen new variant forwards, including replays, reused the verified native arm.

There were **zero sign reversals**, but all twelve absolute changes exceeded both $0.01S$ and the numerical guard $\max(0.005,100r,10^{-4}\max(1,|z_N|,|z_L|))$, where $r$ is the maximum repeated-readout discrepancy. Median absolute change was 460.738083, or $0.854380S$; maximum change was 1037.0191, or $1.923020S$. Both interpolated signs and replay checks remained stable, with $r=0$. This engineering guard is not an error theorem. The result is quantitative sensitivity with qualitative stability on this selected set.

A subsequent immutable-weight audit resolves the public-parameter question: all 90 learned tensors through `mixed4e` are byte-identical, and all 116 main-network/classifier entries match the GraphDef after layout conversion; only auxiliary heads are absent. Reachable repository history contains that corrected blob before Lucent 0.1.8 and Narang's article. This does not verify an author's historical cache or execution. Different preprocessing and redirected gradients also remain distinct. Neither calibration nor the isolated LRN contrast identifies a geometric cause, a shared upstream route, or an error in either semantic interpretation.

## Prospective local-control construction

Three protected construction attempts test whether one author-selected car-front
example can support a local contour intervention with a matched nuisance control.
This is development work on a previously scored source, not an independent test.
The first attempt stopped when its fixed inverse left a 3.736923-pixel residual,
above the 0.05-pixel limit. A separately frozen 30-step bisection inverse reduced
residuals below $7\times10^{-8}$ pixels and let both primary edits pass, but none
of 120 fixed radial shams matched both changed-pixel area and RGB RMS: the largest
RMS ratios were 0.6731125 and 0.6650199. Neither attempt reached visual assessment
or neural inference.

The final admitted family applies the transverse triangular warp
$F_a(x,y)=(x,y+a[y-p(x)]w_p(x,y))$. It fixes the declared parent contour
$y=p(x)$ pointwise and is zero outside the permitted support. Before image access,
a 130-gain rational bank per parent was fixed: paired negative and positive gains
from 0.01 through 0.56, followed by negative gains through 0.74. An exact-hash
static review by Anthropic `claude-opus-5-5`, run at high effort with a 128,000-token
output allowance, returned `GO` with natural `end_turn` and no truncation. The
pre-render self-check passed 611 code and analytic-field
checks while opening and rendering zero images.

The single frozen execution retained all 260 candidates. Twelve source-parent
and fourteen rounded-parent shams met both exact matching intervals. The selected
source sham had changed-area ratio $27/23$ and RMS ratio 0.932919 relative to the
rounding edit; the selected rounded-parent sham had ratios $163/140$ and 0.924290
relative to the rescue edit. Every selected-image preservation check passed, and
all pixels outside the permitted 480-pixel rectangle were identical to the parent.
A separately written saved-artifact replay passed 2,511 checks and recovered both winners
without a new warp.

Numeric feasibility did not pass the frozen model-assessor visual gate. Two isolated role-blind Anthropic
assessors (`claude-opus-5-5` and `claude-sonnet-5-5`, both high effort) inspected
the same fifteen anonymous views. Both classified the nominated contour as
ambiguous in all five prepared $224\times224$ inputs. One kept all 480-level calls
ambiguous; the other described two larger-scale appearance groups but attributed
their separation to seam thickness and contrast, a fender-seam notch and a dark
local tick. Those are nuisance explanations, not a clean contour-only validation.
The frozen rule required a stop on assessor disagreement, ambiguity or visible
control damage. The attempt therefore terminates at
`STOP_VISUAL_AMBIGUITY_OR_CONTROL_DAMAGE`, with zero target/native neural forwards.
The result demonstrates a control-construction limit on this source: exact outside
preservation and matched global pixel magnitude can coexist with an unresolved
model-resolution shape distinction and local visual confounding. This is not a human-perception result.

# Appendix G: Goodfire aggregate and training-export audits

## Original aggregate intervention evidence

The supplied source aggregates answer a different question from prospective SAT prediction. Re-executing the retained arithmetic checks verifies all 345 ablation conditions and 3,795 elementary identities. For the medium ConvNeXt's all-block 75th-percentile cut, the source reports 45,000 examples, 1,429 correct-to-wrong transitions, 970 repairs, and 4,660 prediction changes. Thus net accuracy changes by $100(970-1429)/45000=-1.02$ percentage points while disagreement is $100(4660/45000)=10.3556\%$. The remaining 2,261 changes are wrong-to-different-wrong transitions.

Small net accuracy loss therefore does not establish prediction preservation. It also does not establish that every changed prediction is important or harmful. The original seven-file corpus does not supply the per-example predictions, checkpoints, experiment runners or individual interpretability judgements needed to answer those further questions. The later public-availability check recovered additional training summaries, described below; it did not establish the required native execution access. The 345 conditions reuse examples and are not 345 independent replications. The trained-digit calibration and published-SAT audit are new bounded applications. The Inception calibration reproduces the source illustration signs under its declared preprocessing; it does not independently replicate Goodfire's modern-network experiments.

## What the public training trajectories establish

A further bounded source check on 16 October 2026 recovered measurement exports beyond the supplied seven-file corpus. Goodfire's training map and accompanying fresh-run report describe two 91,828-parameter seeds and one 345,552-parameter run, each observed at sixteen snapshots over ten epochs. These are distinct from the epoch-90 medium and large microscopes. The fresh-run source reports corrected paired image-and-label mixup after a label-only mixup issue in the historical trainer; these runs are not exact replications of that historical recipe. We audited the three runs' metrics, fixed-coordinate trajectories and ablation summaries: 144 coordinate-score rows and 255 candidate/control ablation rows. This is a retrospective audit of exported values, not new neural measurements or independent experimental replication.

The report's onset brackets reproduce under its adjacent-high-sample rule. Broad-car coordinates first sustain AUROC 0.85 between nominal epochs 4–5, 2–3 and 1–2; ambulance coordinates first sustain 0.80 between 5–7, 4–5 and 4–5. No cab coordinate sustains 0.80. These are final discovery-selected coordinates followed backward, not each snapshot's best coordinate. The brackets are sampled-score descriptions, not confidence intervals or continuous emergence times. The final ambulance coordinate lies earlier in network block order than the final broad-car coordinate in all three runs. Temporal score ordering and network-depth ordering therefore answer different questions.

The source separately ablates an early broad candidate upstream of each specialist, against sixteen same-block controls at five times. All three final specialist-response changes and AUROC comparisons fall inside their control ranges. Across all fifteen run/time comparisons, each outcome lies inside its range in fourteen. The exceptions differ: at approximately epoch two, the wider candidate increases oriented specialist response by 0.096503 SD, exceeding the largest control increase of 0.077587 SD; tiny seed one's candidate reduces AUROC by 0.001778, versus a largest control reduction of 0.001422. Other outcomes also have exceptions, including tiny seed zero's final ambulance-margin drop of 0.006795 versus control maximum 0.006547. These signed descriptive differences must not be converted into unsigned damage or significance tests. Control identities change over time, with 46, 47 and 49 distinct controls across the three runs; they are not longitudinal cohorts of sixteen fixed units. Scale matching is not equality: the wider candidate's final contribution scale is 0.04119045, above the control maximum 0.02996745. The written tiny-seed-one final AUROC subtraction is $10^{-16}$ despite equal displayed values; this is retained export-level roundoff, not a material effect.

The initial multi-coordinate readout AUROCs are already 0.855–0.874 at random initialization. Final broad/hard-road AUROCs are 0.861171/0.506636, 0.926448/0.536636 and 0.970159/0.614864; the hard-road comparison has 44 negatives. These source-reported measurements constrain a semantic reading of the broad scores and do not isolate a capacity effect from two tiny seeds and one wider seed.

The inspected report explicitly retains training checkpoints and full probe arrays in its experiment directory. The public exports permit these arithmetic checks, but do not provide native execution or independent recomputation of selectivity from per-image responses. Their evidence supports the reported temporal ordering, with selection and sampling qualifications. It does not establish causal inheritance through a privileged precursor. The tested candidate/control rule also does not rule out distributed or redundant developmental routes. URLs, source hashes, complete signed comparisons and replay code accompany the audit; the availability search is bounded, not a universal nonrelease claim.


The source report already declines a demonstrated consolidation mechanism. This audit verifies its stated summaries and qualifies a broad control-range sentence; it does not newly refute a claim of inheritance that the report itself does not make. The source protocol's stated 20 September freeze is retained as an author-reported date, not authenticated preregistration.

# Appendix H: graph uncertainty and legacy-source corrections

For observed response vectors $\widehat y_i$ with unknown true vectors satisfying $\|y_i-\widehat y_i\|_\infty\le\delta$, the incompatibility graphs at thresholds $2(\varepsilon+\delta)$ and $2(\varepsilon-\delta)$ give necessary and sufficient outer bounds on the number of response labels, when $\delta\le\varepsilon$. Strict inequality defines an edge. These bounds need not jointly be attainable by one perturbation of all vectors.

For the five-cycle example, each of five response coordinates has entries $+3/2$ and $-3/2$ at the corresponding adjacent vertices and zero elsewhere. At $\varepsilon=1$, the exact lower count is three for $\delta<1/2$ and one for $\delta\ge1/2$. The upper count is three for $\delta\le1/4$ and five for $\delta>1/4$. Linear interpolation between sampled counts is incorrect.

![Corrected five-cycle uncertainty bracket. Filled circles include the threshold endpoint; open circles exclude it. The curves are integer step functions. They are outer bounds on the number of labels, not a claim that every count between them is attainable.](figures/c5_corrected.pdf){width=86%}

For a self-contained non-sharpness example, take scalar observations $(0,1,2)$, $\varepsilon=1$ and $\delta=3/4$. The upper graph has threshold $2(\varepsilon-\delta)=1/2$, so it is complete and gives three labels. Every admissible true triple, however, lies in $[-3/4,11/4]$, of width $7/2$. Three pairwise distances greater than two would require a span greater than four. Some pair therefore fits in one radius-one interval, and every realisation needs at most two labels. The triple $(-3/4,1,11/4)$ attains two. Pairwise worst-case edges need not arise from one common perturbation. The released implementation checks non-finite and negative tolerances, rejects a response array with no coordinates, and includes exact rational threshold tests on both sides of each discontinuity. Legacy source files remain unchanged; these corrections apply to this release.


# Appendix I: exploratory and preceding SAT studies

## The first action-selection pilot showed no robust ordering between bits and magnitudes

The first predictive pilot fitted the same histogram-gradient-boosting configuration to each report: 100 iterations, at most 15 leaves, learning rate 0.08, L2 regularisation one, 63 bins and no early stopping. Inputs add 10/40/40 one-hot indicators for slot/old/new clause. Two replacements per fit-origin/slot give 61,440 fit rows; assessment covers all 399,360 edits. Each menu selects the highest predicted UNSAT probability, breaking ties by clause index. All semantic outcomes remain in the denominator.

| Initial report | Response AUROC | Selected edit produces UNSAT |
|:--|---:|---:|
| Action context only | 0.5106 | 11.494% |
| Published 34 bits | 0.9457 | 62.451% |
| Published 34 post-ReLU magnitudes | 0.9405 | 62.266% |
| Fixed random 34 bits | 0.5106 | 11.494% |
| Fixed random 34 magnitudes | 0.5108 | 11.865% |
| Exact current truth table, 32 bits | 0.9630 | 66.963% |

Of 10,240 menus, 8,433 admit an UNSAT-producing edit. Conditional successes are 75.833% for published bits and 75.608% for magnitudes. Magnitudes slightly improve log loss (0.2561 versus 0.2574), but worsen selection by 0.1855 percentage points. At an UNSAT-confidence threshold of 0.5, the two methods act on only 0.098% and 0.586% of menus. The table describes forced-choice ranking, not a calibrated actionable policy. The random threshold codes are all zero, so they reproduce context-only behavior and are not a frequency-matched arbitrary-report control. A post hoc factorial using one learner seed finds a positive magnitude-minus-bits difference only under natural-binary action encoding, while the interval for the menu-by-encoding interaction includes zero. We therefore retain the cell values without claiming a general ordering or encoding effect.

## A frozen negative-score repair failed the task-policy gates

The preceding pilot leaves a specific possibility open: ReLU may erase useful
readiness information by collapsing all negative preactivations to zero, while
positive post-ReLU magnitudes may have supplied the wrong precision in that
first screen. We therefore fixed a second cached development screen before it
accessed the cached outcome arrays.
It uses the same 4,096 origins and contiguous 3,072/1,024 split, but chooses one
slot per origin by SHA-256 from the canonical token bytes and evaluates all 39
legal replacements at that slot. This gives 119,808 fit rows and 39,936
assessment rows. No new transformer forward is used.

The candidate augments each published bit with
$\mathbf 1[\tau_i<s_i<0]$, where $\tau_i$ is the lower median of strictly
negative fit-origin preactivations for unit $i$; the added bit is zero and
$\tau_i=-\infty$ when no such fit value exists. It is a structural refinement of the published report,
so its finite information ceiling cannot be lower; any learned-performance
deficit is a learner/finite-sample result. Six fixed comparators use action
context only, published bits, four preactivation bins, four postactivation bins,
continuous preactivation and continuous postactivation. The preactivation bins
are $(-\infty,\tau_i]$, $(\tau_i,0)$, $[0,0.5)$ and $[0.5,\infty)$. For
postactivation, let $u_i$ be the lower median of fit values at least 0.5, or 0.5
if none exist; the bins are $\{0\}$, $(0,0.5)$, $[0.5,u_i]$ and
$(u_i,\infty)$. All seven models reserve 68 report columns and receive the same
16-bit natural-binary slot/old-clause/new-clause descriptor. They use the same
histogram-gradient-boosting settings as the pilot. All menus remain in every
denominator; ties go to the lower clause ID.

The screen also separates fitted policy performance from report information.
For each report arm, assessment menus are grouped by identical report bytes,
slot and old clause. Within each class, the diagnostic policy selects the new
clause with the largest number of semantic successes. This gives the exact
minimum reachability regret of a deterministic policy on this finite bank. It
is a bank-specific lookup ceiling, not a population or low-complexity decoder
bound. Mean regret is the average of
$\mathbf1[\text{a successful edit is reachable}]-\mathbf1[\text{the selected edit succeeds}]$.
Coverage at threshold $c$ is the fraction of menus whose selected edit
has predicted UNSAT probability at least $c$. At 0.8, success has utility $+1$,
failure $-4$, and abstention zero; the policy acts at that threshold and
otherwise abstains. Five registered learned-policy gates required the candidate
to beat published bits and postactivation four-bin by five percentage points,
remain within two points of preactivation four-bin, attain at least 10%
coverage at 0.8, and have positive mean utility. Only the first three are
distinct comparative tests: zero coverage forces zero abstention utility. Two
further information gates required its exact ceiling to beat the published and
postactivation four-bin ceilings by five points. Report fragmentation made both
unattainable before outcomes were read. Passing required all seven gates, so
this protocol could only stop and should not be read as seven independent
empirical failures. This is a design defect, not merely a negative result: the
earlier pilot had already produced only 0.098% bit-report and 0.586% magnitude
coverage at the lower threshold 0.5, making a 10% requirement at 0.8 foreseeably
unattainable without a different calibration strategy.

| Initial report | Selected edit produces UNSAT | Success when reachable | Mean regret | Coverage at 0.5 / 0.8 | Exact finite ceiling | Singleton fraction among classes |
|:--|---:|---:|---:|---:|---:|---:|
| Context only | 12.598% | 15.176% | 0.7041 | 0% / 0% | 52.832% | 23.243% |
| Published bits | 43.164% | 52.000% | 0.3984 | 0% / 0% | 83.008% | 99.608% |
| Negative bit | 41.406% | 49.882% | 0.4160 | 0% / 0% | 83.008% | 100% |
| Pre four-bin | 44.043% | 53.059% | 0.3896 | 0% / 0% | 83.008% | 100% |
| Post four-bin | 41.016% | 49.412% | 0.4199 | 0% / 0% | 83.008% | 100% |
| Continuous pre | 45.605% | 54.941% | 0.3740 | 0% / 0% | 83.008% | 100% |
| Continuous post | 49.707% | 59.882% | 0.3330 | 0% / 0% | 83.008% | 100% |

Of 1,024 assessment menus, 850 admit a successful edit, so the exact semantic
control reaches 83.008%. The candidate is 1.758 points below published bits,
0.391 points above postactivation four-bin, and 2.637 points below
preactivation four-bin. Every learned arm assigns its selected edit probability
below 0.5; coverage and utility therefore fail together. All seven registered gates
are false, and the branch stops without a protected follow-up. The candidate's
1.758-point deficit to published bits is not an information deficit: the
candidate contains every published bit, so it concerns this fitted learner and
finite development screen.

The ceiling failure needs a different interpretation from the learned failure.
Published bits already induce 1,020 report/context classes for 1,024 menus:
1,016 are singletons and four contain two menus. Occupancy alone guarantees a
ceiling of at least 82.617%; the observed ceiling equals the 83.008% semantic
oracle because each collided reachable class still has a common successful
clause. Every richer report induces only singleton classes on this bank. The candidate cannot
gain five ceiling points because the finite lookup diagnostic has saturated,
not because negative values have been shown information-free. The two ceiling
gates are therefore structurally foreclosed once the published and
postactivation four-bin reports attain the oracle; they are saturation-triggered
stopping rules rather than two independent empirical failures.

The learned rankings remain well below their shared ceiling. An exploratory
saved-data factorial crossed both menu/training regimes with one-hot and
natural-binary action encodings, using one common learner seed and paired
origin-cluster uncertainty. The all-slot/one-hot cell exactly reproduces the
original pilot. With one-hot actions, continuous postactivation differs from
published bits by $-0.001855$ (95% interval $[-0.017480,0.013872]$) in the
all-slot regime and $-0.000977$ ($[-0.020508,0.019531]$) in the selected-slot
regime. With natural-binary actions the differences are $+0.035156$
($[0.013184,0.058694]$) and $+0.065430$ ($[0.037109,0.091797]$).
The encoding effect on the contrast is 3.701 points in the all-slot regime and
6.641 points in the selected-slot regime; the menu-by-encoding interaction is
2.939 points with interval $[-0.011040,0.069438]$. Thus the observed ordering
changes with the action encoding, while the interaction interval does not identify
encoding as the dominant cause. This exploratory factorial
uses the opened bank, fits eight additional cached models and makes no neural
forward. It does not rescue the negative-bin arm or establish a population
ordering between bits and magnitudes.

## A fitted readiness bit predicts named native units after paired edits

A separate development contract then targets information erased by ReLU. The singleton-assignment rules in Palumbo et al.'s supplement (their Appendix G) nominate 17 units covering 16 valuations:

| Unit(s) | Assignment | Unit | Assignment |
|:--|:--|:--|:--|
| 10 | TFFFF | 261 | TTFTT |
| 41 | FFFFF | 291 | TFTTT |
| 55,416 | TTTFF | 317 | TFFFT |
| 77 | FFTTT | 326 | TTTTF |
| 81 | FTTTF | 380 | TFFTT |
| 185 | FFFTT | 411 | TTFFT |
| 224 | FFFFT | 492 | TFTFT |
| 245 | FTFFT | 495 | FFFTF |

These are approximate published hypotheses, not reassigned labels. For valuation $v$, let $q_v(\varphi)$ count violated clauses. On original fit cases satisfying $q_v\ge1$ and $s_i<0$, fit $r_i=\mathbf1[s_i>\rho_i]$ to distinguish $q_v=1$ from $q_v\ge2$. Minimise balanced error over observed float32 thresholds and constant rules; ties favour fewer positives, then the largest threshold. A missing fit class gives constant zero. The direction is fixed, and $\rho_i$ is never recalibrated using edited outcomes. All these original postactivations are exactly zero.

Before extraction, select 32 origins per unit, split and q stratum by a fixed SHA-256 ordering using only formula counts. Choose a false old clause, keep one false literal, and replace the other variable by a distinct variable absent from that clause. The paired replacements use its true or false literal under $v$: repair reduces $q_v$ by one; neutral leaves it unchanged. The action choice is hash-determined, never activation-selected. Clipped eligibility is evaluated afterwards without replenishment.

The exact semantic update and frozen native forecast are

$$m_v(a)=\mathbf1[v\not\models c]-\mathbf1[v\not\models d],\qquad
q_v(\varphi^a)=q_v(\varphi)-m_v(a),\qquad
\widehat b_i(\varphi^a)=r_i(\varphi)\,\mathbf1[m_v(a)=1].$$

The target is the same native unit's actual post-edit event $b_i$, not SAT itself. Counts enter selection, threshold supervision and evaluation; the predictor receives only its readiness bit and declared action. The assessment retains 955 clipped unit-origin pairs from 1,088 selected pairs, with both actions. Duplicate TTTFF units are averaged first, then the 16 assignments equally.

Negative-score AUROC is 0.944201. Native balanced accuracy is 0.909543; the mean repair-minus-neutral difference in activation-event rates for $q_v=1$ exceeds that for $q_v\ge2$ by 0.809763. All 16 assignment interactions are positive; three individual pairs reverse direction and remain included. Two thousand origin-cluster bootstrap draws, over 685 original IDs, yield descriptive intervals [0.890265,0.927831] for balanced accuracy and [0.770677,0.846883] for the interaction. These condition on fitted rules and the opened cohort, not independent units or protected data.

| Native-event forecast | Assignment-macro balanced accuracy |
|:--|---:|
| Frozen readiness and true action effect | 0.909543 |
| Semantic action effect only, $\mathbf1[m_v(a)=1]$ | 0.787105 |
| Post-ReLU score, constant by clipped-case construction | 0.500000 |
| Cyclically mismatched unit readiness | 0.587205 |
| Correct readiness, cyclically mismatched assignment effect | 0.529938 |
| Action-only logistic probe | 0.621549 |
| Symbolic $q_v=1$ readiness oracle | 0.956180 |
| Subsequent continuous-score comparator | 0.892684 |

The semantic-only rule predicts activation after the declared repair and
inactivity after the neutral edit, without a neural state readout. Adding the
fitted readiness bit improves assignment-macro balanced accuracy by 0.122438;
the paired 2,000-draw original-origin interval is
$[0.104108,0.137696]$. This isolates incremental predictive value for the
readiness bit on this opened cohort. The symbolic $q_v=1$ oracle reaches
0.956180, so the fitted neural rule closes 72.4% of the 0.169075 gap between
semantic action effect alone and that oracle while remaining 0.046637 below it.
This makes the violation-count oracle, rather than the unequal-supervision
continuous fit, the primary interpretive comparator. On the 133 assessment pairs excluded by
the frozen clipped-case estimand, both rules reach 0.953432 and their difference
is zero; over all 1,088 selected assessment pairs the corresponding values are
0.910856 and 0.789392. These rows were not removed from the archive or silently
converted into primary cases.

Mismatches cyclically shift the increasing unit-ID list. The action probe uses the same 90 context indicators, balanced classes and L2 logistic regression ($C=1$). The symbolic oracle is neither a neural feature nor an upper bound on native prediction. The post-ReLU arm is constant because primary cases were selected with negative preactivation and hence zero postactivation; it is a construction check, not a discriminating comparator. Missing classes propagate NA; intervals require 1,900 valid draws without replenishment. No unit is dropped. The initial screen required 20 assessment cases per clipped q class/unit, macro-AUROC 0.65 and 12 assignments above 0.60. Native transport required ten per class, balanced accuracy 0.70, interaction 0.25, twelve positive interactions, and gaps of 0.10 against post/mismatch controls and 0.05 against context. All passed, although the post-arm gap is automatic under the clipped-case construction. These are development screens, not mechanism-identification criteria.

An exploratory cached-data comparator fitted continuous-score boosted trees after the readiness result. It scores 0.892684 balanced accuracy, versus 0.909543 for readiness, but uses only 1,922 native unit-action labels against 45,693 semantic unit-origin labels and reaches 0.990741 fit accuracy. The mismatch in supervision and likely overfitting prevent an interpretable comparison, so this model supplies no main-text claim. Its complete contract, models, saved probabilities, paired interval and replay remain in the research package.

A known-unit query stores one readiness bit versus 32 continuous-score bits. Across 17 units this is 17 versus 544 bits; retaining the original 34-bit report gives 51 versus 578 state bits. The common action costs 16 fixed bits (4/6/6 for slot/old/new), giving 67 versus 594. The earlier 34-magnitude report costs 1,088 bits. Readiness additionally stores 544 threshold bits, 34 rule-tag bits, a 238-bit unit/assignment table and the shared 1,280-bit clause lookup. The action-only probes use 1,547 binary64 coefficients. Continuous models contain 3,400 trees, 16,778 leaves and 13,378 splits; serialized estimators total 2,182,704 bytes. Both require native state extraction. Full ordered formulas cost 60 fixed or 54 packed bits; semantic multisets require $\lceil\log_2\binom{49}{10}\rceil=33$ packed bits plus decoding/SAT computation, but do not preserve sequence-sensitive native outputs. These reports are not globally minimal SAT codes.

The retained `sat_case`, `sat_mechanism` and `sat_margin_baseline` artifacts contain the contracts, complete action tables and executable checks. The prediction stages retain their unsuccessful comparisons and stop without extending their assessment sets; their execution and review scopes are reported below.

The result adds quantitative prospective performance to source-fixed assignment hypotheses. Generic negative-margin information and the source's weighted-count/threshold rationale substantially anticipate its direction (Palumbo et al., 2025; Heo et al., 2019; Deng et al., 2026; Ameisen et al., 2025; Swain and Dutta, 2026). Since readiness is a deterministic function of the continuous score, a sufficiently informed continuous decoder can reproduce it. No unique mechanism or context invariance follows. The FTTTT assignment made satisfying by the original witness edit is outside the singleton set: this study does not repair that witness, all 34 units, or global action selection. Nor does a negative value currently mediate output through its zeroed ReLU.

## Preceding four-direction matched-edit pilot

Before the all-unit study, we froze a response-blind stress test of two source-nominated assignment pairs under consequential edits and one variable swap. It remains useful because its paired origins and edited endpoints support the later direct-readout intervention accounting. It is superseded by the all-unit study for general label-validity claims. The tested pairs are units 261/291, interpreted as assignments TTFTT/TFTTT, and 411/492, interpreted as TTFFT/TFTFT. They are the only two pairs among the 17 published singleton-assignment units that are closed under the chosen swap, so no further pair was selected by model response. Assignment characters are ordered by variable index. Swapping $x_1$ and $x_2$ exchanges the two members of each pair; write this renaming as $\pi$.

The construction samples 100,000 ordered ten-clause formulas from the 36 native clause types that exclude the variable pair $\{x_1,x_2\}$. This exclusion preserves literal positions under the adjacent-variable swap. For each pair $(a,b)$, a positive origin $G$ is UNSAT, violates exactly one clause under $a$, and at least two under $b$. An edit retains one old literal that is false under $a$ and true under $b$, and replaces the other literal by a different variable's literal that is true under $a$. We require $G'$ to have sole satisfying assignment $a$. Thus $\pi G'$ has sole satisfying assignment $b$. A matched negative origin $H$ has at least two violations under both assignments and remains UNSAT after the identical edit descriptor: slot, old clause, retained literal, replacement variable and new clause.

A deterministic hash order selects at most one edit per origin and then 32 distinct descriptors per pair, each with one positive and one negative origin. Both cohorts retain their complete quartets $(G,G',\pi G,\pi G')$. All 512 endpoint token sequences are distinct and excluded against the 8,957 unique sequences in the complete earlier native-query ledger. An independent Boolean checker verifies all semantic conditions, exclusions and selection constraints before inference. A later study-module-free replay independently regenerates the complete 100,000-proposal PCG64 stream, semantic eligibility, action multiplicities, domain-separated hash order, prior-ledger exclusions and all 64 selected blocks. Its 16 array/metadata comparisons agree elementwise, including every selected trial, action, endpoint and expected bit; the replay used NumPy 2.4.2 while the frozen construction used 2.4.6. This verifies deterministic construction identity conditional on the named generator, not external preregistration or representativeness. Selection uses no model confidence, activations or task outcomes. The finite target is the equal-weight mixture of these selected descriptors in one checkpoint, not the source distribution or a population of units.

For the primary pair, the expected positive bit vectors are $(0,0),(1,0),(0,0),(0,1)$, at the source threshold 0.5. Every primary bit in the negative cohort and every off-pair bit should be zero. The fixed benchmark requires, separately for each pair, at least 58/64 positive bits, 404/448 negative bits, 26/32 complete positive eight-bit patterns and at most 51/512 off-pair false positives. A separate directional diagnostic requires at least 29/32 true positives and 202/224 true negatives; a localized renaming failure requires the original direction to pass and the renamed direction to fail. These tolerances are our finite-cohort diagnostic choices, not guarantees stated by Palumbo et al. We also compare the four-bit correct mapping $P=(261\;291)(411\;492)$ with identity and the wrong mapping $W=(261\;492)(291\;411)$. On positive edited endpoints, each error is Hamming discrepancy divided by four; the advantages are $e_I-e_P$ and $e_W-e_P$. Both pair-specific advantages and the pooled lower bootstrap endpoints must be positive. All-zero controls are not pooled into that decisive contrast.

The combined benchmark is **not met**:

| Pair | Positive bits | Negative bits | Exact positive patterns | Off-pair false positives | Correct-map advantage over identity / wrong map |
|---|---:|---:|---:|---:|---:|
| 261/291 | 63/64 | 448/448 | 31/32 | 0/512 | 0.484375 / 0.484375 |
| 411/492 | 56/64 | 448/448 | 24/32 | 0/512 | 0.375000 / 0.375000 |

On this sole-solution pilot cohort, the first pair has 63/64 sensitivity and 31/32 complete patterns. The later all-unit bank does not support a general threshold-fidelity claim for that pair: unit 261 detects only 23/32 multi-solution positives there. The second pilot pair misses the positive-bit and exact-pattern cutoffs. Its unit-411 sole-solution deficit, 24/32 here, recurs at 23/32 in the all-unit bank; unit 492 on $\pi G'$ reaches 32/32 in the pilot. The labels “original” and “renamed” denote construction roles under an involution, not a privileged causal direction. Renaming supplies matched inputs here; the observed deficit occurs in the designated original direction and is not a failure after renaming.

The 224 negative target bits per direction comprise 192/192 true negatives on UNSAT inputs and 32/32 on SAT inputs whose different sole solution excludes the labelled assignment. Each off-pair denominator similarly comprises 384/384 UNSAT and 128/128 different-sole-solution true negatives. No multi-solution SAT negative is present. “Perfect specificity” on the pooled denominator would therefore obscure that most negatives are UNSAT and that the harder multi-solution regime is untested.

A post hoc zero-forward threshold audit separates ranking from source-threshold fidelity:

| Unit and construction direction | TP at 0.5 | FP at 0.5 | Preactivation AUROC | TP at zero FP | FP at 32/32 TP |
|:--|--:|--:|--:|--:|--:|
| 261, original | 31/32 | 0/224 | 1.000000 | 32/32 | 0/224 |
| 291, renamed | 32/32 | 0/224 | 1.000000 | 32/32 | 0/224 |
| 411, original | 24/32 | 0/224 | 0.999302 | 31/32 | 5/224 |
| 492, renamed | 32/32 | 0/224 | 1.000000 | 32/32 | 0/224 |

Unit 411's weakest positive preactivation is -0.004926, while its largest negative is 0.278990. Its eight misses are therefore not merely eight values immediately below 0.5: one positive overlaps the negative range. Lowering the threshold can recover seven misses without a false positive; recovering all eight produces five false positives. The result is unequal source-threshold sensitivity plus one failure of threshold separability, rather than absence of the ranked direction. Exact architectural equivariance is not tested: the learned token embeddings and checkpoint are not constrained to implement the variable swap.

The unit-411/unit-492 threshold counts differ by 8-to-0 on the matched descriptors. We report this descriptively: the comparison involves different units on transformed inputs, and the architecture does not imply equality as a null hypothesis. The predeclared localised-failure pattern—adequate sensitivity in the designated original direction followed by inadequate sensitivity after renaming—is absent. The correct-mapping error is descriptively lower than identity and the wrong mapping, with pooled advantage 0.429688 and interval [0.390625,0.468750], but those controls coincide because off-pair bits are zero and add no independent corroboration.

The model classifies 505/512 endpoints correctly: 254/256 for the first pair and 251/256 for the second, with pooled balanced accuracy 0.985677. All sixteen output tokens participate in argmax, and no non-answer wins. Always predicting UNSAT would achieve 75% ordinary accuracy. Task failures remain in every unit denominator. The two original and renamed directions, their controls, and all outcomes travel together in 2,000 saved matched-block bootstrap draws, stratified by pair. These are conditional descriptive intervals; the selection and matching do not establish independent sampling or population coverage. Heuristic intervals of radius $10^{-4}+10^{-5}|v|$ leave every bit, task class and benchmark decision unambiguous. They are sensitivity ranges, not rigorous floating-point error bounds. The 512 initial and sixteen replay input forwards agree exactly on the two replayed blocks; no learner is fitted.

## Downstream contribution is a separate causal question

The checkpoint's final MLP feeds a linear unembedding without final normalisation. With post-ReLU vector $h$, pre-MLP residual $r$ and $C=W_{\rm out}^{\mathsf T}W_U$, its real-arithmetic readout is
$$
 z=rW_U+b_{\rm out}W_U+hC.
$$
The predeclared secondary calculation reconstructs all sixteen logits from saved native float32 tensors in float64; the largest discrepancy is $1.348\times10^{-5}$, within the declared parity tolerance. Let $z_{\rm edit}$ denote this reconstructed readout at the edited endpoint. Restoring only pair $Q$ to its corresponding origin values, with all other edited-state variables fixed, gives
$$
 z_{\rm clamp}=z_{\rm edit}-(h_{{\rm edit},Q}-h_{{\rm origin},Q})C_Q.
$$
The same calculation uses the other fixed pair as an off-pair control and retains both cohorts and both directions. This is an **origin-restoration** intervention, not zeroing. The operations differ on 12/128 positive endpoints because at least one nominated origin value is nonzero (4/64 for pair 261/291 and 8/64 for pair 411/492). The other fixed pair is a source-fixed control, not an activity- or contribution-matched control.

The nominated restoration changes 115/128 positive edited decisions, all from correct to wrong. Conditioning on the native decision gives the more relevant denominator: 115 of 126 initially correct positive endpoints become wrong. The fixed off-pair restoration changes none, and neither fixed restoration changes any of the 128 negative edited decisions. These controls are near-automatic under the construction: only 7/128 negative endpoints have any nominated-pair activation change, and only 11/128 positive endpoints have any off-pair activation change. All class changes remain resolved under the declared sensitivity ranges. Mean nominated-pair SAT-minus-UNSAT contribution is 8.741825 logit units in the positive cohort, versus 0.024632 for the fixed off-pair control; corresponding negative-cohort means are 0.075333 and 0.007434.

A post hoc single-unit patch-in supplies the converse direct-readout diagnostic. For active source unit $u$, it adds the edit-associated change to the matching UNSAT origin while leaving its residual stream and every other unit at the origin value,
$$
z_{\rm patch}=z_{\rm origin}+(h_{{\rm edit},u}-h_{{\rm origin},u})C_u.
$$
The full sixteen-way argmax changes from UNSAT to SAT on 29/32, 32/32, 18/32 and 32/32 cases for units 261, 291, 411 and 492, respectively: 111/128 in total. This is sufficiency at the saved final linear readout cut, not an on-manifold network-level sufficiency intervention.

We then select activation-profile comparisons using only each unit's sorted 32-case absolute activation-change profile. A qualified match has relative Euclidean profile distance at most 0.25, fixed before reading its output effect. Only unit 374 qualifies for unit 261: restoration changes 1 rather than 29 decisions, and patch-in changes 4 rather than 29. Only unit 201 qualifies for unit 411: the corresponding counts are 10 versus 22 and 6 versus 18. These are not semantic nulls. Source unit 374 responds to TTFTT, the assignment examined in the first original direction, and unit 201's rule includes TTFFT, the assignment examined in the second. Neither renamed direction has a qualified profile match; the source changes are too isolated for the requested comparison. The wider 32-nearest sets are retained as references but are not described as matched.

For completeness, a direction-specific post hoc reference evaluates every single unit and unordered pair on each 32-endpoint direction. The active source unit has the largest correct-to-wrong count in every direction:

| Pair and direction | Initially correct | Active unit | Correct→wrong | Rank among 512 units | Best non-pair unit | Best pair excluding both nominated units |
|:--|--:|--:|--:|--:|:--|:--|
| 261/291, original | 32/32 | 261 | 29 | 1 | unit 374: 1 | 185+374: 2 |
| 261/291, renamed | 32/32 | 291 | 32 | 1 | none: 0 | none: 0 |
| 411/492, original | 30/32 | 411 | 22 | 1 | unit 201: 10 | 36+201: 13 |
| 411/492, renamed | 32/32 | 492 | 32 | 1 | none: 0 | none: 0 |

The rank is mechanically favoured by the sparse construction: many units do not change at all, and adding an inactive unit creates hundreds of tied pair interventions. On the 411 direction, unit 201 is the more informative same-assignment comparison. Its source rule is TTFFT AND NOT TFTFF, so it is a more conditional TTFFT detector rather than a task-irrelevant control. Unit 201 is active on 4/8 unit-411 threshold misses and 17/24 hits. Restoring it changes 6/8 missed-threshold decisions and 4/24 hit decisions from correct to wrong. This is consistent with distributed compensation for some misses, while neither proving identical semantics nor identifying a unique route. Unit 411 itself changes four missed-threshold decisions and leaves four unchanged under restoration.

On these sole-solution cohorts, source-unit restoration and patch-in produce large source-aligned direct-logit effects while the stricter bit-pattern benchmark remains unmet. Because the final MLP enters a linear unembedding with no later normalisation, the saved-tensor calculation is the exact intervention effect at that cut up to the reported reconstruction rounding. Both directions of intervention can be off the natural activation manifold; only two original directions admit a close activation-profile comparison and neither is a semantic null; no multi-solution cohort is tested; and the calculation says nothing about the upstream route. Per-unit source-threshold fidelity and direct-readout contribution therefore remain distinct findings.


![Fresh SAT renaming study and secondary downstream accounting. Left: the construction-labelled directions for each fixed pair, with denominator 32; the dashed line is the predeclared directional threshold of 29/32. Pair 411/492 shows an 8-to-0 source-threshold sensitivity asymmetry. This directional threshold is distinct from the combined 58/64 and 26/32 benchmark criteria. Right: correct-to-wrong changes under nominated-pair and other-pair saved-tensor origin restoration, conditioned on initially correct endpoints (126 positive and 128 negative). The direction-specific null and near-automatic control counts are reported in the text. The restorations are float64 calculations from saved native arrays at the exact linear readout cut.](figures/sat-renaming-summary.pdf){width=100%}


# Appendix J: evidence lineage and supporting replay scopes

Public native replay additionally requires separately retrieved, hash-pinned Palumbo upstream source files. The inherited `model.py`, `plot.py` and `modular_addition_utils.py` are not redistributed because an applicable grant from their original upstream source has not been established. Supply these public-source files using `reproduce.py --upstream-directory`; the launcher checks their hashes before staging them into its disposable workspace. Eighteen runnable public stages therefore do not mean that all model code is self-contained in the archive. Historical descriptions of retained source holdings below concern the original private input; public dependency instructions and manifests govern the derivative.

The private supplied v4.7 archive has the 23-stage launcher replayed in the main paper: twelve regenerated-JSON comparisons, one full protected-bank native execution and ten supporting exact/saved-data checks. Statistics are recomputed on archived arrays after the full native equivalence check. Runtime package snapshots, anchored original freeze hashes, field-level provenance-path comparisons and a claim-to-artifact map accompany that private archive. The public derivative has eighteen runnable stages and omits five image-dependent historical stages for redistribution-rights reasons. Descriptions below of complete image holdings refer to private retained inputs, not public availability. These guards do not supply OS-level isolation, fresh training, a second-host replication, renewed early pilot/witness forwards or renewed human/model assessment. Superseded numerical exports remain historical; `effective_state_followup_v6.json` is the corrected learned-basis/readout record, and `nmf_revision_diagnostic.json` gives the subsequent per-assignment and separability checks. The prior eight-row checks and version-2 manifest counts below describe earlier verification rather than the public derivative stage inventory.



The retained research package contains model weights, dataset and splits, initial and supplementary contracts, freeze manifests, integer response banks, fitted report parameters, exact witnesses, source code, literature ledger, and review records. This manuscript must be distributed with those materials and the all-unit SAT, SAT-renaming and Goodfire components; the paper alone is not a completed evidence release. The complete methods and proofs needed to interpret the results appear in this manuscript. Data files provide executable detail rather than missing premises. The standard-library checker `certificates/verify_exact.py` validates the retained contrast witnesses without a numerical solver. NumPy reproduces the models and finite response tables; SciPy is used to propose LP solutions; exact rational arithmetic checks the scientific certificate boundary.

The all-unit SAT freeze manifest has SHA-256 prefix `77ac6f5f3c41abce`, and the selected bank has prefix `38a8980436b67d7a`. Native inference loaded the checkpoint strictly under Python 3.11.14, NumPy 1.26.4 and Torch 2.1.2, evaluated 7,252 unique inputs with two CPU threads, and produced output prefix `20c3a3dbb0c5a161`; the historical eight-row fresh-load replay agrees exactly. The corrected release now additionally re-executes all 7,252 protected inputs and records bitwise agreement of all retained arrays on this host. Analysis under the separately pinned Python 3.14.7 and NumPy 2.4.2 environment produced prefix `d81470a4b98a0b04`. Post hoc diagnostic versions 2 and 3 have prefixes `91536f14c7663f04` and `191cf677d6a66e0e`; the rendered all-unit PDF figure has prefix `93d5ae6986088853`. The full-layer study has frozen-manifest prefix `4662155b6d3e4d73`, initial extraction prefix `94a27ab24bd57baa`, eight-row replay prefix `100846c57353d45b` and protected analysis prefix `4927369087d06a70`. Its preflight and self-checks passed 3,227 assertions before native execution. The zero-forward localisation diagnostic has prefix `2376633ac87df382`; the effective-state analysis v2 has prefix `b837c1447167432c` and binds its superseded indexing-error v1 by hash; the zero-forward weight, threshold-substitution, all-unit competition, task-error and standardised-rotation follow-up has prefix `4aeaaaf9be56882b`; the zero-forward purity, learned-basis, refitted-readout and joint-zero follow-up has prefix `25a1ab870aa87701`; the final sparse-report PDF figure has prefix `6c7c8ac6c6f16d4b`. The source notebook containing the unit-nomination rule has prefix `5410cc7cc28e28d3`. The extraction covers all 512 units and preserves the earlier seventeen columns and logits bitwise. The PDF-build receipt binds the protected SAT studies, manuscript, cited figures and named diagnostic scripts; older supporting components have their own manifests. A post-analysis linkage audit checks base and transformed proposal links, exact token unions, unit inventories, all 9,469 historical rows and the native/replay hash chain. Model-generated reviews are retained as criticism and provenance, not described as human peer review or independent replication.

Historical verification had distinct scopes: the original audit matched all 55 files in the then-supplied version-2 manifest. The previous aggregate analysis and 486 diagnostic assertions were rerun. The supplied thirty Torch decoder fits were not rerun for this corrected release; they are legacy optimiser controls and are not evidence for the new contribution. The primary study includes three fresh classifiers, frozen held-out evaluations, exact contrast replay, and precision-bound checks; the nonlinear extension adds three exploratory classifiers. The supporting theory controls compare primal and separately written dual programmes and check small repair problems against enumeration. None of this is human peer review or formal proof-assistant verification. The published-SAT checkpoint, retained query banks, native arrays, fitted parameters and source mapping accompany the prospective studies. Their saved-result audits are separate from re-executing the transformer. The readiness study used 8,434 forwards with separate-process replay. Its independent numerical/selection audit contains 19,922 assertions; the cached comparator audit contains 5,407 and does not refit its trees or independently regenerate probabilities. The negative-score screen was admitted only after 52 source/synthetic checks and a three-tree compatibility smoke replayed byte-identically. Its single frozen execution fitted seven 100-tree models to cached outcomes, made zero native forwards and stopped after every development gate failed. A separate read-only audit recomputed the gates and verified the terminal--manifest--output hash chain without decoding saved models or research inputs. The static admission deliberately did not inspect cache semantics; runtime checks bind array shapes, Boolean outcome dtype, clause range and unit order, but do not explicitly require a dtype for native pre/post arrays. Report/log closeout corrections changed assembly or inventories only, preserving frozen scientific inputs and arrays. The Inception supplement includes the canonical GraphDef, exact image inputs and all 28 native/variant forward outputs; its portable NumPy checker re-evaluates readouts, gates, saved replays and contribution arithmetic without executing TensorFlow. Public-weight identity and historical execution are explicitly distinguished.

Eight later diagnostics use only retained SAT arrays and are marked exploratory. The finite-bank information audit groups all legal edits by exact declared lookup input, reproduces the archived aggregates, re-denominates conflicts over non-singleton classes and refines published classes with the truth table. A companion sensitivity analysis repeats that decomposition on deterministic nested banks and by report activity. The semantic-only readiness calculation first reproduces the frozen structured-rule bootstrap exactly, then applies the same 2,000 saved assessment-origin resamples to the declared semantic baseline. The action-menu factorial makes eight fixed-seed tree fits and reproduces the original all-slot/one-hot pilot exactly before crossing the two menu regimes with two encodings. The threshold audit reports all four units' score distributions and exact pairwise AUROCs. The restoration audits count nonzero origin values, stratify specificity by SAT semantics, perform single-unit source patch-in, select controls from activation-change profiles and test the published unit-201 conjunction as a compensation candidate. The direction-specific null ranks all 512 single units and 130,816 pairs separately on each 32-endpoint direction. Together they make zero new transformer forwards, propose no new formula, and do not convert the opened development bank into protected confirmation.

A separate construction replay addresses a different provenance question. Without importing the study protocol, sampler, constructor or semantic oracle, it reimplements the five-variable grammar and ordinary Boolean evaluation, draws the declared 100,000 PCG64 proposals, replays every eligibility and hash-order decision, and reproduces the complete selected bank. All 16 stored-array and metadata comparisons pass, covering candidate clauses and counts, eligibility codes, action multiplicities, eligible-record order, selected trials, endpoint tokens, truth counts, task labels and expected unit bits. The replay makes zero neural forwards and fits no learner. It strengthens the deterministic selection record; it does not authenticate an external timestamp, establish that these formulas were absent from checkpoint training, or turn the conditional descriptor mixture into a random unit or formula sample.

The SAT-renaming attempt first failed while importing Torch because resolving the virtual-environment interpreter symlink selected the base interpreter. It never entered native execution: checkpoint loads, attempted batches and charged forwards were zero. The failed attempt and receipts remain preserved. A separately recorded recovery retained the lexical interpreter path, copied 25 frozen implementation/data artifacts byte-identically, and changed no scientific code, selected input, threshold or proposal. The original deadline was retained. The recovery completed the same 512 initial and 16 replay input forwards, with zero fits or additional proposals; the initial failure is not retrospectively relabelled as successful. A separately written semantic-and-arithmetic checker and a separate execution-record audit cover the recovered result. The former's author also wrote the pre-query Boolean bank verifier. These are checks within the same research programme, not blind reviews, independent retraining or new native experimental replications. That saved-data audit did not regenerate the proposal pool; the later study-module-free construction replay closes this narrower gap while remaining same-programme computational verification.

The Goodfire audit reconciles nine measurement exports, 144 fixed-coordinate scores, 255 ablation rows and all 120 signed candidate/control field comparisons. Its original 206 assertions and same-agent 1,368-check rational replay are distinct from a separately written standard-library reconstruction, which passed 29,543 assertions. These counts describe software checks, not scientific sample sizes. Portable numerical-only replay has a narrower source-provenance scope, which its receipt states explicitly. Source-reported discovery selection, control matching and protocol dates are preserved as author statements rather than independently authenticated procedures. The portable `sat_renaming_case/` and `goodfire_training_audit/` components contain the evidence, original contracts and scoped replay instructions; their archived reviews retain the exact input identities and limitations.

The reproduction package preserves three deterministic construction attempts as separate components. The second retains every candidate image, exact matching record and primary field array. Its saved-data checker and a separately written pixel/integer audit reproduce the matching failure without new warps or neural inference. Rejected radial candidates retain scalar geometry summaries, not complete fields, so their extrema are not independently remeasured by those audits. The third retains all 260 fixed-seam candidates, an exact winning-selection replay, a complete high-effort Opus 5.5 static review, and two isolated role-blind visual reports. Its saved checker passes 2,511 assertions without rerendering. The visual stop is external to the immutable numeric run and binds both reports by hash. None of these construction records supplies a neural causal result.


# Appendix K: response-blind source-grammar audit and native agreement

## Sampling and chronology

The new bank contains 4,096 formulas, with 2,048 SAT and 2,048 UNSAT, generated with seed 2026101801. Following the source clause grammar, each formula contains ten distinct non-tautological clauses over five variables; each clause uses two distinct variable identities, giving forty possible clauses. Formula-level clause multisets are deduplicated while clause order is retained for native evaluation. This differs from the protected engineered proposal bank's sampling with replacement. Balancing, deduplication and exclusion make the new bank history-excluded and source-grammar-based, not literal IID sampling at natural prevalence. No signed-variable-permutation orbit exclusion is claimed.

Before any new-bank model inference, a history scan found that the initial v1 bank overlapped unselected historical clause-only proposals. V1 was preserved and never forwarded. An expanded inventory of 51 arrays contained 365,753 distinct clause multisets. V2 used the unchanged seed and class quotas: 5,158 draws, four historical/duplicate rejections and 1,058 class-full rejections yielded the bank. This was an input-only repair, not response-based selection. Unknown training/test corpora remain unavailable, so history exclusion does not prove training absence.

The checkpoint, seventeen source-nominated labels and threshold 0.5 were frozen unchanged. Assignment indexing was reconciled explicitly: the generator uses least-significant-bit-first variable order whereas the protected label map uses most-significant-bit-first order. Sixteen batches covered all 4,096 rows; repeating the first 256-row batch gave bitwise equality. There were no non-answer full-vocabulary argmaxes. Native task accuracy was 4,087/4,096, comprising 2,042/2,048 SAT and 2,045/2,048 UNSAT. Assignment-macro AUROC was 0.9974772794362474, weighting the duplicated assignment once.

| Unit | AUROC | Positive detections / positives | False positives / negatives |
|---:|---:|---:|---:|
| 10 | 0.997499 | 117/121 | 53/3975 |
| 41 | 0.999097 | 134/137 | 21/3959 |
| 55 | 0.993246 | 83/113 | 34/3983 |
| 77 | 0.997990 | 135/138 | 39/3958 |
| 81 | 0.991974 | 88/125 | 50/3971 |
| 185 | 1.000000 | 121/121 | 0/3975 |
| 224 | 1.000000 | 132/132 | 0/3964 |
| 245 | 0.998354 | 123/125 | 40/3971 |
| 261 | 0.999090 | 102/110 | 14/3986 |
| 291 | 0.999208 | 120/122 | 20/3974 |
| 317 | 1.000000 | 96/98 | 0/3998 |
| 326 | 0.998470 | 101/104 | 23/3992 |
| 380 | 1.000000 | 115/116 | 0/3980 |
| 411 | 0.989887 | 93/116 | 53/3980 |
| 416 | 0.990519 | 90/113 | 60/3983 |
| 492 | 0.996679 | 111/112 | 72/3984 |
| 495 | 0.999507 | 153/157 | 19/3939 |

All 85 unit-by-semantic-stratum cells are retained in the audit receipt. Original protected quota-based gates were not reapplied to these new denominators. Independently implemented Boolean evaluation, rank-sum AUROC and threshold recounts passed in both normal Python and actual optimized mode. These checks support the new numerical result, not a new checkpoint or independent-host replication. The native environment was Python 3.11.14, NumPy 1.26.4 and Torch 2.1.2, CPU strict loading, two native threads and one BLAS thread.

## Retrospective native agreement of the original-bank truth classifiers

The freshly replayed v6 nested classifiers retain their complete error-row sets relative to binary Boolean truth. With exactly two classes and no abstention, each prediction equals truth XOR membership in that error set. Index domain, uniqueness and error counts were checked before reconstruction. Comparing these predictions against the original-bank native full-sixteen-token argmax gives:

| Original-bank decoder | Boolean truth errors | Native argmax agreement |
|:--|--:|--:|
| Standardised threshold bits | 64/7252 | 7205/7252 (99.3519%) |
| Standardised continuous state | 26/7252 | 7221/7252 (99.5725%) |
| Unscaled threshold bits | 64/7252 | 7205/7252 (99.3519%) |
| Unscaled continuous state | 21/7252 | 7224/7252 (99.6139%) |

Native Boolean truth accuracy on that bank is 7,215/7,252. Derived agreement is a distinct retrospective quantity, not a native-target fit, logit reconstruction, recovered model coefficients or fresh-bank deployment. No fitted truth classifier was evaluated on the new bank. Exact disagreement indices are retained with the reconstruction receipt.

## Frozen identities and replay scope

The v2 bank SHA-256 is `f2d30d0f09dab11c6e06859f4e8b1630a652eb77a5e1d7bfc02e9e3706bd0e13`; the pre-inference evaluation freeze is `9ee46c4c3392cc9d2eb4520b0c23bae43ffbe838c9ce0f858555a066225bf434`. Fresh native arrays have SHA-256 `ed534712e0451cc76b0e9f0bdf24ae4ec39949495fc1cbf00fdee7501f3645dd`. These bind the new evidence separately from the original protected freezes.

The supplied private archive passed all 23 stages in normal mode, including bitwise reproduction of all 7,252 protected native rows. The historical runner rejects `-O` explicitly; separate assertion-independent headline and new-bank checks passed under actual `-O`. This is not a claim that all historical scientific checks ran optimized. The public derivative exposes 18 runnable stages; five image-dependent historical stages are not distributed because underlying photograph rights remain unresolved. Their negative or stopped outcomes remain described in this supplement. Earlier statements about complete image packages describe retained private historical inputs, not public availability. Python audit hooks constrain top-level stage Python file access but do not automatically propagate to nested subprocesses and are not an operating-system sandbox.

# Appendix L: excluded training-feasibility attempts

Independent-checkpoint evidence would require competent models trained from fresh random initialisations, followed by a separately specified mechanistic evaluation. We tested only whether a small local development budget could supply such models. None of the attempts below is a confirmatory scientific replicate, and no representation or mechanism-bank results were inspected.

| Seed | Regimen and status | Presentations | Final validation accuracy |
|---:|:--|--:|--:|
| 999001 | Incorrect grammar allowed repeated variables, repeated clauses and tautologies; interrupted and excluded | 380,000 retained | Not a valid source-task feasibility endpoint |
| 999002 | Correct distinct-clause grammar; reduced corpus/batch, constant learning rate | 1,000,000 | 63.10% |
| 999003 | Correct grammar; reduced corpus/batch, source linear warmup restored | 1,000,000 | 70.475%; failed frozen 99% criterion |

The valid pilots used 20,000 balanced training formulas and 4,000 balanced validation formulas, disjoint by canonical clause multiset, with exhaustive 32-assignment truth labels and an independent scalar check of 200 labels. Each used fresh initialisation and seeded Python, NumPy and Torch. Source architecture was retained: two layers, heads [1,4], model dimension 128 and MLP width 512. Effective batches of approximately 128, the smaller corpus and the one-million-presentation cap differ from source-faithful training; these are explicitly different-regimen pilots.

Seed 999002 completed in 466.20 seconds. Final training accuracy was 62.805%; validation SAT and UNSAT accuracies were 51.10% and 75.10%, respectively, with no non-answer predictions. Seed 999003 completed in 518.94 seconds, with final training accuracy 71.205%; validation SAT and UNSAT accuracies were 76.25% and 64.70%, respectively, again without non-answer predictions. Its source AdamW warmup used an initial zero learning rate and epoch-wise `min(epoch/10,1)`, reaching 0.001 after ten epochs; betas were (0.9,0.98) and weight decay one. The prospective protocol fixed the final-checkpoint rule, at most one million presentations and a 600-second check at epoch boundaries, and required at least 99% final full-vocabulary validation accuracy. That criterion failed. No best checkpoint or further repair was selected.

Warmup cannot be credited causally for the accuracy difference because the authorised fresh seed and data also changed. Failure under these short regimens does not establish failure under source-faithful training. Reading the source training log corrected an initial filename-based cost estimate: the historical data contain 1,000,000 SAT plus 1,000,001 UNSAT examples, with approximately 1.2 million training rows per epoch, so epoch 900 represents approximately 1.08 billion presentations. Short synthetic throughput estimates suggest roughly 85--105 training hours per source-scale model on this host before substantial overhead; these are projections, not measured full-run times. No thirty-model campaign was launched.

The final warmup protocol SHA-256 is `1a0d69ce17924be10aaca84f10ffafc7d188f0c01916f3295e70588ffc83597d`; its original code hash is `0f686b99c3d8c7d6e9e12b6ada369342ed2557f613b165baac39afcaddfe2fb6`. The final data and checkpoint hashes are `61b8b4bddc301c94e8482d241d9586ff0289b050f5e56e333ef127d2521f5c98` and `deac4d48e1501d39cf6b5e55f6cded3ec995d2bcd7e76dc927d173705ced55cb`. Curves and task-only endpoint records preserve the failure. All pilot seeds, data and checkpoints remain excluded from any future confirmatory sample. The stopping outcome preserves, rather than resolves, the single-checkpoint scientific boundary.

# References {.unnumbered}

Alpaydin, E., & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C50P49>. CC BY 4.0; the study uses the retained 1,797-image scikit-learn subset.

Antverg, O., & Belinkov, Y. (2022). On the pitfalls of analyzing individual neurons in language models. *NAACL*, 2602--2628. <https://arxiv.org/abs/2110.07483>.

Aytug, H. (2015). Feature selection for support vector machines using generalized Benders decomposition. *European Journal of Operational Research, 244*, 210–218. <https://doi.org/10.1016/j.ejor.2015.01.006>.

Basu, S., Seo, D., & Varshney, L. R. (2022). Hypergraph-based source codes for function computation under maximal distortion. *IEEE Journal on Selected Areas in Information Theory, 3*(4), 824–838. <https://doi.org/10.1109/JSAIT.2022.3232222>. Inspected full-text version: <https://arxiv.org/abs/2204.02586v2>.

Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017). Network Dissection: Quantifying interpretability of deep visual representations. *IEEE Conference on Computer Vision and Pattern Recognition*, 6541--6549. <https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html>.

Bertsimas, D., & Van Parys, B. (2020). Sparse high-dimensional regression: Exact scalable algorithms and phase transitions. *The Annals of Statistics, 48*(1), 300–323. <https://doi.org/10.1214/18-AOS1804>. Author manuscript: <https://arxiv.org/abs/1709.10029>.

Bills, S., et al. (2023). *Language models can explain neurons in language models*. OpenAI. <https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html>.

Bolukbasi, T., Pearce, A., Yuan, A., Coenen, A., Reif, E., Viégas, F., & Wattenberg, M. (2021). *An interpretability illusion for BERT*. <https://arxiv.org/abs/2104.07143>.

Bricken, T., et al. (2023). *Towards monosemanticity: Decomposing language models with dictionary learning*. Transformer Circuits Thread. <https://transformer-circuits.pub/2023/monosemantic-features/index.html>.

Cammarata, N. (source snapshot supplied; accessed 15 October 2026). *Neural Networks Are Already Clean*. Goodfire. <https://already-clean.goodfire.com/>.

Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., et al. (2022). *Causal scrubbing: A method for rigorously testing interpretability hypotheses*. <https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing>.

Bhatt, R. (2026, October 15). *What a write becomes: the descendant and quotient program on Weight-Dictionary Decomposition*. Unrefereed public experiment repository; recursive-generalisation and SwiGLU follow-up experiments e671–e679. Inspected commit `1626b9b4e3ca`. <https://github.com/sunmoonron/wdd-descendants/tree/1626b9b4e3caebae7a9ea2e2db55d947d48fdde9>.

Yandell, J. (2026, October 15–17). *Already-clean field-net audit and formation study*. Unrefereed public experiment records for a finite card-play policy. Inspected commit `a313e38f70db`. <https://github.com/jasonyandell/mk5-main/tree/a313e38f70db3ae2f6dcfbe0b7d5a30a4763d708/wiki/experiments>.


Chughtai, B., Chan, L., & Nanda, N. (2023). A toy model of universality: Reverse engineering how networks learn group operations. *Proceedings of Machine Learning Research, 202*, 6243--6267. <https://proceedings.mlr.press/v202/chughtai23a.html>.

Chanin, D. (2026). *Are sparse autoencoder benchmarks reliable?* Preprint. <https://arxiv.org/abs/2605.18229>.

Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *Transactions of the Association for Computational Linguistics, 9*, 160--175. <https://doi.org/10.1162/tacl_a_00359>.

Elhage, N., Nanda, N., Olsson, C., et al. (2021). *A mathematical framework for transformer circuits*. Transformer Circuits Thread. <https://transformer-circuits.pub/2021/framework/index.html>.

Elhage, N., Hume, T., Olsson, C., et al. (2022). *Toy models of superposition*. Transformer Circuits Thread. <https://transformer-circuits.pub/2022/toy_model/index.html>.

Elhage, N., Lasenby, R., & Olah, C. (2023). *Privileged bases in the Transformer residual stream*. Transformer Circuits Thread. <https://transformer-circuits.pub/2023/privileged-basis/index.html>.

Elboher, Y., Gottschlich, J., & Katz, G. (2020). An abstraction-based framework for neural network verification. *Computer Aided Verification*. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7363186/>.

Geiger, A., Ibeling, D., Zur, A., et al. (2025). Causal abstraction: A theoretical foundation for mechanistic interpretability. *Journal of Machine Learning Research, 26*(83), 1–64. <https://jmlr.org/papers/v26/23-0058.html>.

Geiger, A., Lu, H., Icard, T., & Potts, C. (2022). Causal abstractions of neural networks. <https://arxiv.org/abs/2106.02997>.

Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. D. (2023). *Finding alignments between interpretable causal variables and distributed neural representations*. <https://arxiv.org/abs/2303.02536>.

Geva, M., Caciularu, A., Wang, K. R., & Goldberg, Y. (2022). Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 30--45. <https://doi.org/10.18653/v1/2022.emnlp-main.3>.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of Machine Learning Research, 70*, 1321--1330. <https://proceedings.mlr.press/v70/guo17a.html>.

Gurnee, W., Nanda, N., Pauly, M., Harvey, K., Troitskii, D., & Bertsimas, D. (2023). *Finding neurons in a haystack: Case studies with sparse probing*. <https://arxiv.org/abs/2305.01610>.

Gurnee, W., Horsley, T., Guo, Z. C., Rezaei Kheirkhah, T., Sun, Q., Hathaway, W., Nanda, N., & Bertsimas, D. (2024). *Universal neurons in GPT-2 language models*. <https://arxiv.org/abs/2401.12181>.

Hase, P., Bansal, M., Kim, B., & Ghandeharioun, A. (2023). Does localization inform editing? Surprising differences in causality-based localization vs. knowledge editing in language models. *Advances in Neural Information Processing Systems, 36*. <https://arxiv.org/abs/2301.04213v2>.

Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP-IJCNLP*, 2733--2743. <https://doi.org/10.18653/v1/D19-1275>.

Hewitt, J., Ethayarajh, K., Liang, P., & Manning, C. D. (2021). Conditional probing: Measuring usable information beyond a baseline. *EMNLP*, 1626--1639. <https://doi.org/10.18653/v1/2021.emnlp-main.122>.

Heimersheim, S., & Nanda, N. (2024). *How to use and interpret activation patching*. <https://arxiv.org/abs/2404.15255>.

Huang, J., Wu, Z., Potts, C., Geva, M., & Geiger, A. (2024). RAVEL: Evaluating interpretability methods on disentangling language model representations. *ACL*, 8669–8687. <https://doi.org/10.18653/v1/2024.acl-long.470>.

Huang, J., Geiger, A., D'Oosterlinck, K., Wu, Z., & Potts, C. (2023). Rigorously assessing natural language explanations of neurons. *BlackboxNLP*, 317–331. <https://doi.org/10.18653/v1/2023.blackboxnlp-1.24>.

Hilaire, T., Ouzia, H., & Lopez, B. (2019). Optimal word-length allocation for the fixed-point implementation of linear filters and controllers. *26th IEEE Symposium on Computer Arithmetic*, 175–182. <https://doi.org/10.1109/ARITH.2019.00040>.

Karvonen, A., Rager, C., Lin, J., et al. (2025). SAEBench: A comprehensive benchmark for sparse autoencoders in language model interpretability. *Proceedings of Machine Learning Research, 267*, 29223–29264. <https://proceedings.mlr.press/v267/karvonen25a.html>.

Lin, D., Talathi, S., & Annapureddy, S. (2016). Fixed point quantization of deep convolutional networks. *Proceedings of Machine Learning Research, 48*, 2849–2858. <https://proceedings.mlr.press/v48/linb16.html>.

Lecomte, V., Thaman, K., Schaeffer, R., Bashkansky, N., Chow, T., & Koyejo, S. (2024). *What causes polysemanticity? An alternative origin story of mixed selectivity from incidental causes*. <https://arxiv.org/abs/2312.03096>.

Li, M., & Janson, L. (2024). *Optimal ablation for interpretability*. *Advances in Neural Information Processing Systems, 37*. <https://arxiv.org/abs/2409.09951>.

Li, L., Walsh, T. J., & Littman, M. L. (2006). Towards a unified theory of state abstraction for MDPs. *International Symposium on Artificial Intelligence and Mathematics*. <https://thomasjwalsh.net/pub/aima06Towards.pdf>.

Littman, M. L., Sutton, R. S., & Singh, S. (2002). Predictive representations of state. *Advances in Neural Information Processing Systems, 14*, 1555--1561. <https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf>.

Makelov, A., Lange, G., Geiger, A., & Nanda, N. (2024). Is this the subspace you are looking for? An interpretability illusion for subspace activation patching. *International Conference on Learning Representations*. <https://arxiv.org/abs/2311.17030>.

McGrath, T., Rahtz, M., Kramár, J., Mikulik, V., & Legg, S. (2023). *The Hydra Effect: Emergent self-repair in language model computations*. <https://arxiv.org/abs/2307.15771>.

Mu, J., & Andreas, J. (2020). Compositional explanations of neurons. *Advances in Neural Information Processing Systems, 33*, 17153--17163. <https://proceedings.neurips.cc/paper/2020/hash/c74956ffb38ba48ed6ce977af6727275-Abstract.html>.

Olah, C., Cammarata, N., Voss, C., Schubert, L., & Goh, G. (2020). Naturally occurring equivariance in neural networks. *Distill, 5*(12). <https://doi.org/10.23915/distill.00024.004>.

Palumbo, N., Mangal, R., Wang, Z., Vijayakumar, S., Păsăreanu, C. S., & Jha, S. (2025). Validating mechanistic interpretations: An axiomatic approach. *Proceedings of Machine Learning Research, 267*, 47509–47544. <https://proceedings.mlr.press/v267/palumbo25a.html>. Inspected appendices: <https://arxiv.org/html/2407.13594v2>. Author artifact: <https://github.com/nilspalumbo/axiomatic-validation/tree/35d1e4105265ab0ca74bc7daf9371fc8f82bba81>.

Pan, L., Ganesh, V., Abernethy, J., Esposo, C., & Lee, W. (2025). Can transformers reason logically? A study in SAT solving. *Proceedings of Machine Learning Research, 267*, 47632--47671. <https://proceedings.mlr.press/v267/pan25d.html>.

Pimentel, T., Valvoda, J., Maudslay, R. H., Zmigrod, R., Williams, A., & Cotterell, R. (2020). Information-theoretic probing for linguistic structure. *ACL*, 4609--4622. <https://doi.org/10.18653/v1/2020.acl-main.420>.

Ravichander, A., Belinkov, Y., & Hovy, E. (2021). Probing the probing paradigm: Does probing accuracy entail task relevance? *EACL*, 3363--3377. <https://arxiv.org/abs/2005.00719>.

Petreczky, M., Bako, L., & van Schuppen, J. H. (2013). Realization theory of discrete-time linear switched systems. *Automatica, 49*(11), 3337–3344. Author manuscript: <https://arxiv.org/abs/1103.1343>.

Vafa, K., Chen, J. Y., Rambachan, A., Kleinberg, J., & Mullainathan, S. (2024). Evaluating the world model implicit in a generative model. *Advances in Neural Information Processing Systems, 37*. <https://arxiv.org/abs/2406.03689v3>.

Méloux, M., et al. (2026). *Validating causal abstraction metrics on simulated complex systems*. Preprint. <https://arxiv.org/abs/2607.00267>.

Shafran, O., Geiger, A., & Geva, M. (2026). *Constructing interpretable features from compositional neuron groups*. Version 2. <https://arxiv.org/abs/2506.10920v2>.

Sutter, D., Minder, J., Hofmann, T., & Pimentel, T. (2025). The non-linear representation dilemma: Is causal abstraction enough for mechanistic interpretability? *NeurIPS 2025*. <https://arxiv.org/abs/2507.08802v2>.

Selsam, D., Lamm, M., Bünz, B., Liang, P., de Moura, L., & Dill, D. L. (2019). Learning a SAT solver from single-bit supervision. *International Conference on Learning Representations*. <https://arxiv.org/abs/1802.03685>.

Sajjad, H., Durrani, N., & Dalvi, F. (2022). Neuron-level interpretation of deep NLP models: A survey. *Transactions of the Association for Computational Linguistics, 10*, 1285--1303. <https://doi.org/10.1162/tacl_a_00519>.

Scherlis, A., Sachan, K., Jermyn, A. S., Benton, J., & Shlegeris, B. (2022). *Polysemanticity and capacity in neural networks*. <https://arxiv.org/abs/2210.01892>.

UCI Machine Learning Repository. Optical recognition of handwritten digits. <https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits>.


Voita, E., Ferrando, J., & Nalmpantis, C. (2023). *Neurons in large language models: Dead, n-gram, positional*. <https://arxiv.org/abs/2309.04827>.

Voita, E., & Titov, I. (2020). Information-theoretic probing with minimum description length. *EMNLP*. <https://aclanthology.org/2020.emnlp-main.14/>.

Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. *International Conference on Learning Representations*. <https://arxiv.org/abs/2211.00593>.

Yang, L., Zhang, L.-H., & Zhang, Y.-N. (2023). *The $L_q$-weighted dual programming of the linear Chebyshev approximation and an interior-point method*. <https://arxiv.org/abs/2308.07636>.

Xu, Y., Zhao, S., Song, J., Stewart, R., & Ermon, S. (2020). A theory of usable information under computational constraints. *International Conference on Learning Representations*. <https://openreview.net/forum?id=r1eBeyHFDH>.

Zhang, F., & Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. *International Conference on Learning Representations*. <https://openreview.net/forum?id=Hf17y6u9BC>.


Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The Clock and the Pizza: Two stories in mechanistic explanation of neural networks. *Advances in Neural Information Processing Systems, 36*. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/56cbfbf49937a0873d451343ddc8c57d-Abstract-Conference.html>.

Zhou, J., & Klindt, D. (2026). *Complete or Sparse: A Tale of Two Identifiabilities*. Version 1. <https://arxiv.org/abs/2610.20851v1>. Recorded source-access date: 18 October 2026.

Ma, H., Mei, Z., & Sala, F. (2026). *Behavioral Compositionality of Sparse Autoencoder Representations*. Version 1. <https://arxiv.org/abs/2610.21261v1>. Recorded source-access date: 18 October 2026.

Zavattari, C., Tommasi, A., & Prencipe, G. (2026). *Human Interventions as Labels: Localizing, Steering, and (Sometimes) Editing the Computations Behind LLM Agent Failures*. Version 1. <https://arxiv.org/abs/2610.21708v1>. Recorded source-access date: 18 October 2026.

Gu, J., Zhang, H., Chen, C., & Aleti, A. (2026). *Lexical Reference Frames: Interpreting LM Residual Streams in Native Semantics*. Version 1. <https://arxiv.org/abs/2610.22169v1>. Recorded source-access date: 18 October 2026.

Ameisen, E., et al. (2025). *Circuit Tracing: Revealing Computational Graphs in Language Models*. Transformer Circuits. <https://transformer-circuits.pub/2025/attribution-graphs/methods.html>.

Deng, B., Wan, Y., Yang, B., Huang, F., Wang, W., & Feng, F. (2026). *SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs*. ICLR 2026; inspected version 2, 28 February 2026, original preprint 2025. <https://arxiv.org/abs/2507.14894v2>.

Heo, B., Lee, M., Yun, S., & Choi, J. Y. (2019). Knowledge transfer via distillation of activation boundaries formed by hidden neurons. *AAAI, 33*(1), 3779--3787. <https://doi.org/10.1609/aaai.v33i01.33013779>.


Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *International Conference on Learning Representations*. <https://openreview.net/forum?id=9XFSbDPmdW>.

Narang, H. (2026, July 12). *Is that cat? A fox? Or a car? Disentangling the famous neuron mixed4e:55 in InceptionV1*. Author research article. <https://pages.narang99.in/posts/2026-07-12-disentangling-mixed4e-55/>. Inspected companion code: <https://github.com/narang99/hiccup-ide/tree/f63722930a81d9f2aaa1f5d621ba30060f2d4547/olt>.

Phoenix, J. (2026, October 16). *Are neural networks already clean? An independent, automated test on Qwen3-8B*. Version 1; unrefereed public research report. Inspected commit `6cc33cc8fb0e`. <https://github.com/jphoenix-dev/ModelMapper/tree/6cc33cc8fb0e1d95b97bf9aecb049b4ee1955c6b>.

DongerBot. (2026, October 15). *clean-nets: Testing claims from Neural Networks Are Already Clean*. Unrefereed public experiment repository. Inspected commit `6b54ac97ec4d`. <https://github.com/dongerbot/clean-nets/tree/6b54ac97ec4d5a4bd26c79c57130f8c2b303b0e7>.

Long, T. (2026, October 15). *already-clean-check: Independent, preregistered tests of claims in Neural Networks Are Already Clean*. Unrefereed public experiment repository. Inspected commit `0cf42ec17e15`. <https://github.com/TheoHLong/already-clean-check/tree/0cf42ec17e155ad46c8feded55478af436778d8b>.

Eiras, F. (2026, October 17). *On the Polysemanticity of the V Neuron: Code, pre-registrations and results*. Unrefereed public experiment repository. Inspected commit `6b0ea14a8796`. <https://github.com/fgirbal/v-neuron-test/tree/6b0ea14a879617c674f8ea713443d844bcbe49d8>.

Li, J. (2026, October 17). *Do weak MLP activations carry important information?* Unrefereed public experiment repository. Inspected commit `d4abef8da037`. <https://github.com/lijuliana/weak-activation-ablation/tree/d4abef8da0376c328685a43ad02beff0433e72da>.

Mittal, A. (2026, October 16). *Are Neural Networks Already Clean? A small replication*. Unrefereed public experiment repository. Inspected commit `715c62b6f908`. <https://github.com/AMX07/already-clean-replication/tree/715c62b6f9080f2e2d52ed815a4534be187502b3>.

Boix, E. (2026, October 16). *random_is_clean: Are random neuron combinations as interpretable as neurons?* Unrefereed public experiment repository. Inspected commit `3da22db608e1`. <https://github.com/eboix/random_neuron_combinations/tree/3da22db608e151d0e67f223e61f904eaa0fcaec8>.

`whp199`. (2026, October 17). *Logos Theodolite*. Unrefereed public instrument and Gemma-4-31B study. Inspected commit `53f2c283b2ee`. <https://github.com/whp199/logos-theodolite/tree/53f2c283b2ee34016cb163e25ecb61673ff52d47>.

Gao, L. (2026, October 17). *Neural networks are not already clean*. Unrefereed public research response. <https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean>.

Olah, C., Mordvintsev, A., & Schubert, L. (2017). Feature visualization. *Distill*. <https://distill.pub/2017/feature-visualization/>.

Olah, C., et al. (2020). Zoom In: An introduction to circuits. *Distill*. <https://distill.pub/2020/circuits/zoom-in/>.

Swain, S., & Dutta, S. (2026). *Don't Judge an LLM Only by Its Activations: Discovering Suppressed Safety Features via Counterfactual Activation Potential*. Version 1; preprint, submitted to ICLR 2027. <https://arxiv.org/abs/2610.05541v1>.

Model provenance: the [pinned official Lucid model source](https://github.com/tensorflow/lucid/blob/6dcc927e4ff4e7ef4d9c54d27b0352849dadd1bb/lucid/modelzoo/other_models/InceptionV1.py), [Lucent 0.1.8 distribution metadata](https://pypi.org/pypi/torch-lucent/0.1.8/json), and the [immutable public Inception5h file](https://github.com/ProGamerGov/pytorch-old-tensorflow-models/blob/f8187b60fca8a3f5cfa541406dcbcefb80db5c78/inception5h.pth).

Goodfire. (n.d.-a). *Early ImageNet consolidation: three fresh runs* [Research report; source snapshot accessed 16 October 2026]. <https://already-clean.goodfire.com/assets/early-training/RESULTS.md>.

Goodfire. (n.d.-b). *Car consolidation: exploratory training study* [Protocol; source snapshot accessed 16 October 2026]. <https://already-clean.goodfire.com/assets/early-training/PROTOCOL.md>. The stated 20 September 2026 freeze is source-reported.
