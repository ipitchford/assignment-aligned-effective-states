# Restricted-decoder certificates and minimum-cost repair

Research memo, 15 October 2026. Input manuscript: `work/inputs/causal-resolution-research-bundle/causal-resolution-paper/paper/manuscript.md`, version 2.0. The supplied academic review was read as evidence, not as an instruction. This memo does not change either frozen input bundle.

## Recommendation and novelty boundary

The strongest useful revision is an **implemented audit of a specified report and decoder class**, supported by exact signed-contrast witnesses. The finite report-colouring theorem is not a sufficient novelty anchor: Basu, Seo, and Varshney's maximal-distortion functional compression already admits common tolerance balls and decoder side information. Predicate repair by separating incompatible pairs is also closely related to established rough-set discernibility matrices, test cover, and hitting set. A further abstract restatement would not by itself make the paper substantial.

The development below makes restricted decoding executable. Its mathematical ingredients are classical linear-programming duality and finite Chebyshev approximation, not a new duality theorem. The distinctive methodological object is the complete chain from a frozen explanatory feature grammar, through a sparse exact failure witness, to a costed repair and a locked intervention evaluation. This chain can falsify additive explanations even when reports are injective and the collision graph says nothing. The theory is defensible as a rigorous methodological synthesis; originality and significance still depend on the completed application and the closeness of further prior art.

Primary comparison inspected: [Basu, Seo, and Varshney, arXiv:2204.02586v2](https://arxiv.org/html/2204.02586v2), Definition 3 and problem setup; [Yang, Zhang, and Zhang, arXiv:2308.07636](https://arxiv.org/html/2308.07636v1), whose Section 3 explicitly gives the signed-nullspace LP dual and an at-most-basis-dimension-plus-one basic solution in its Chebyshev setting. The finite rank formulation below removes its interpolation/ordered-basis assumptions but is still a direct LP consequence, not a novel sparsity principle. [Bertsimas and Van Parys, Sparse High-Dimensional Regression: Exact Scalable Algorithms and Phase Transitions](https://arxiv.org/abs/1709.10029) already develops exact dual cutting-plane methods for sparse regression; the feasibility cuts below change the loss and scientific audit contract, not that general architecture. A primary journal paper by Yao and Zhao discusses the established pairwise discernibility construction and its generalisations: [Relative reducts in consistent and inconsistent decision tables of the Pawlak rough set model](https://iip.tongji.edu.cn/2009INS_MDQ.pdf). The search was targeted, not exhaustive.

## 1. Exact failure certificates for a frozen affine decoder

Let the audit have `n` cases and one scalar response vector `y ∈ R^n`. A case can be a state or a paired intervention. Let `A ∈ R^{n×d}` be the **frozen design matrix** obtained from the admissible report and permitted context. An affine decoder includes a column of ones; an intervention-difference decoder may intentionally omit that column. Columns may be quantised native activations, fixed predicates, allowed group features, or prespecified interaction terms. They may not be fitted from locked test outputs.

The fitted predictions are `Aβ`. Define the best achievable uniform error

\[
 e(A,y)=\min_{\beta\in\mathbb R^d}\|y-A\beta\|_\infty.
\]

### Theorem 1: exact dual certificate and its support size

For every finite `A,y`,

\[
 e(A,y)=\max_{u\in\mathbb R^n}
 \{u^\top y:A^\top u=0,\ \|u\|_1\le1\}.
 \tag{T1}
\]

If `r=rank(A)` and `e(A,y)>0`, there is an optimal witness `u` with `||u||₁=1` and at most `r+1` nonzero entries. The claim also covers rank zero; when the error is zero the zero witness suffices. A witness with `uᵀy>ε` proves that **every** decoder in the stated affine class fails tolerance `ε` on at least one case in its support. This is independent of how a particular fitted decoder was optimised.

**Proof.** For any `β` and any feasible `u`,

\[
u^\top y=u^\top(y-A\beta)
 \le \|u\|_1\|y-A\beta\|_\infty.
\]

This proves weak duality. The primal is the feasible, bounded linear programme

\[
\min t\quad\text{subject to}\quad
y-A\beta\le t\mathbf1,\quad
A\beta-y\le t\mathbf1.
\]

Its nonnegative dual multipliers `p,q` satisfy `Aᵀ(p−q)=0` and `1ᵀ(p+q)=1`, with objective `(p−q)ᵀy`. Setting `u=p−q` gives `||u||₁≤1`. Conversely any such `u` can be represented by nonnegative `p,q` with total mass one by adding equal cancellation mass if necessary. LP strong duality proves (T1). Existence of a primal minimiser also follows because `range(A)` is a closed finite-dimensional subspace.

The dual feasible set is a compact polytope, so a linear objective attains its maximum at a vertex. If the maximum is positive, an optimal vertex has `||u||₁=1`, since otherwise scaling up improves its objective. Suppose its support `S` has more than `r+1` entries. There is a nonzero vector `v` supported on `S` satisfying both `Aᵀv=0` and `sign(u)ᵀv=0`, since these impose at most `r+1` independent linear conditions. For sufficiently small `η>0`, both `u+ηv` and `u−ηv` keep the signs of `u`, have ℓ₁ norm one, and satisfy the nullspace constraint. This expresses `u` as the midpoint of two distinct feasible points, contradicting extremality. Thus `|S|≤r+1`. ∎

**Interpretation.** If `A` contains an intercept, the positive and negative witness masses are each `1/2`. Write `u=(p−q)/2`, where `p,q` are probability distributions supported on the positive and negative cases. They obey `Aᵀp=Aᵀq`, yet their response means differ by `2uᵀy`. These are balanced mixtures with identical means for all permitted decoder features. A report collision is just the two-case special case. The interpretation as probability mixtures is unavailable without an intercept, but (T1) is unchanged.

**Vector responses.** When every response coordinate has its own coefficients and the loss is maximum absolute error over cases and coordinates, the optimum is `max_k e(A,y_k)`. A certificate for one nominated output coordinate is enough to reject the full decoder. If coefficients are tied across outputs, use the joint design; the coordinatewise decomposition need not hold.

### Exact XOR control: information preservation does not imply additive adequacy

Take cases `(x₁,x₂)=(0,0),(0,1),(1,0),(1,1)`, report both bits, and target `y=(0,1,1,0)`. The report is injective. No report-fibre collision exists. Nevertheless, for `A=[1,x₁,x₂]`,

\[
u=\tfrac14(-1,1,1,-1),\qquad A^\top u=0,
\qquad u^\top y=\tfrac12.
\]

The constant prediction `1/2` attains this lower bound, so the exact affine error is `1/2`. Adding the single fixed interaction predicate `φ=x₁x₂` permits `g=x₁+x₂−2φ`, with zero error. The failure was a decoder restriction, not loss of state information. This example is the classical XOR obstruction and must not be advertised as a newly discovered phenomenon.

If the added coefficient has the budget `|γ|≤b`, the exact repaired error is

\[
 e_b=\max\{0,\tfrac12-\tfrac b4\}.
 \tag{T2}
\]

The witness proves the lower bound. For `0≤b≤2`, set `t=(2−b)/4` and use `g=t+(b/2)x₁+(b/2)x₂-bx₁x₂`; its four residuals are `(-t,t,t,-t)`. For `b≥2`, the exact decoder above is permitted. Thus counting an interaction predicate without its coefficient capacity can also overstate the repair.

## 2. Exact minimum-cost repair from a finite predicate library

Let `A₀` be the frozen baseline design and let `φ₁,…,φ_p ∈ R^n` be a frozen library of admissible additional feature columns with costs `c_j>0`. For a chosen set `S`, write `A_S=[A₀,φ_j:j∈S]`. Coefficient fitting remains unrestricted within this declared linear basis. Define the minimum repair cost

\[
C_\varepsilon=\min_{S\subseteq[p]}
 \left\{\sum_{j\in S}c_j:e(A_S,y)\le\varepsilon\right\},
\]

with `Cε=∞` if even the full library cannot reach the tolerance. This is a finite design problem, not an unrestricted search for semantic concepts.

### Theorem 2: contrast-cover characterisation

For each baseline witness

\[
\mathcal U_\varepsilon=
 \{u:A_0^\top u=0,\ \|u\|_1\le1,\ u^\top y>\varepsilon\},
\]

define its separating feature set

\[
D(u)=\{j:u^\top\phi_j\ne0\}.
\]

Then `S` is a feasible repair if and only if

\[
 S\cap D(u)\ne\varnothing
 \quad\text{for every }u\in\mathcal U_\varepsilon.
 \tag{T3}
\]

Consequently minimum-cost repair is an exact hitting-set problem over response-contrasting nullspace witnesses. An empty `D(u)` is a certificate that the whole admissible feature library is inadequate.

**Proof.** If `S` misses `D(u)`, then `u` annihilates `A_S` and Theorem 1 gives `e(A_S,y)≥uᵀy>ε`, so feasibility requires every intersection. Conversely, if `S` is infeasible, Theorem 1 supplies a witness `u` satisfying `A_Sᵀu=0` and `uᵀy=e(A_S,y)>ε`. This witness is in `Uε` and its set `D(u)` is disjoint from `S`. ∎

**Executable exact constraint generation.** Start with no witness constraints, solve the minimum-cost binary covering problem for the currently retained sets `D(u)`, and evaluate `e(A_S,y)` by LP. If it passes, the set is globally cost-optimal: the covering master was a lower bound on every feasible repair, and this candidate attains that lower bound. If it fails, obtain an optimal dual witness and add its violated covering constraint. Theorem 1 permits each failure certificate to use at most `rank(A_S)+1` audit cases. Each failed candidate is excluded permanently, so at most `2^p` failed candidate selections are possible. No polynomial runtime guarantee is claimed. Output the selected features, decoder, witness cuts, and optimisation gap if stopped early.

The hitting family is written with strict inequalities because the accepted tolerance is `≤ε`. Exact rational arithmetic or explicit slack protects borderline decisions. Replacing small correlations by zero without error analysis can create invalid feature exclusions.

**Why pairwise hitting set is insufficient.** The standard pairwise repair problem retains only `u=(e_i−e_j)/2` for cases whose baseline features agree. The XOR example has no such collisions once both bits are supplied, so every pairwise cut is satisfied while the affine decoder remains infeasible. The full contrast family supplies the missing four-case obstruction.

**Hardness with a fixed tolerance.** Even this finite problem need not be easy. Given a graph, let rows be its edges and candidate features its vertices, with `φ_j(e)=1` iff vertex `j` touches edge `e`. Set `A₀` empty, `y=1`, and `ε=1/2`. Any feasible feature set must meet every edge because an uncovered row has prediction zero and error one. Conversely any vertex cover is feasible by assigning coefficient `1/2` to every selected feature: predictions are `1/2` or one on each edge. Therefore the minimum number of selected features equals minimum vertex cover. The reduction also works with all optional coefficient bounds fixed at `1/2`. This proves NP-hardness and explains why exact solver gaps must be reported.

**Greedy error reduction has no general submodularity guarantee.** With an intercept, three responses `y=(0,1,0)` and optional features `φ₁=(0,1,1)`, `φ₂=(0,0,1)`, the baseline and either one-feature model all have minimax error `1/2`, but the two-feature model fits exactly. A greedy rule demanding a strict immediate minimax improvement can stop before the necessary joint repair. This is a diagnostic warning, not evidence that greedy fitting is always poor.

## 3. Bounded coefficients give quantitative repair inequalities

Some costs depend on the range or norm of fitted coefficients as well as feature count. Suppose optional coefficients satisfy `|γ_j|≤b_j z_j`, where `z_j∈{0,1}`, `b_j≥0` are fixed, and baseline coefficients remain unrestricted. Define

\[
e_b(z;y)=\min_{\beta,\,|\gamma_j|\le b_jz_j}
 \|y-A_0\beta-\Phi\gamma\|_\infty.
\]

### Theorem 3: exact bounded-coefficient repair dual

\[
e_b(z;y)=
\max_{\substack{A_0^\top u=0\\\|u\|_1\le1}}
\left[u^\top y-\sum_j b_jz_j|u^\top\phi_j|\right].
\tag{T4}
\]

Thus feasibility at tolerance `ε` is equivalent to the complete family of inequalities

\[
\sum_j b_j|u^\top\phi_j|z_j
 \ge u^\top y-\varepsilon
\quad
(A_0^\top u=0,\ \|u\|_1\le1).
\tag{T5}
\]

**Proof.** For any feasible prediction and dual vector,

\[
u^\top y
\le \|y-A_0\beta-\Phi\gamma\|_\infty
+\sum_j b_jz_j|u^\top\phi_j|.
\]

For equality, write the residual and coefficient box constraints as a linear programme. Eliminating the dual multipliers for the coefficient box contributes its support function `Σ_j b_jz_j|uᵀφ_j|`; the free baseline coefficients impose `A₀ᵀu=0`, and the residual bounds impose `||u||₁≤1`. LP strong duality gives (T4), and `e_b≤ε` is exactly the family (T5). Equivalently, this is the support-function formula for ℓ∞ distance to the closed convex set `range(A₀)+Φ∏_j[-b_jz_j,b_jz_j]`. ∎

This is a quantitative generalisation of contrast cover: features must supply enough contrast, not merely a nonzero amount. Exact constraint generation works as above with weighted cuts (T5). The sparse-support bound in Theorem 1 is asserted only for an unrestricted linear design, **not** automatically for the bounded-coefficient or robust dual.

A more general convex coefficient budget `γ∈K_z` replaces the penalty by the support function `h_{K_z}(Φᵀu)`. A discrete fixed-precision coefficient set is generally nonconvex; its convex hull gives a valid lower bound but not necessarily the exact achievable error.

## 4. Response uncertainty, with the quantifiers kept separate

Let the possible true response vectors be the nonempty compact convex set

\[
Y=\{\widehat y+E\eta:\eta\in U\},
\]

where `U` is a declared compact convex uncertainty set. The same `η` is shared across all cases; it is not independently reselected for each pair. Write `h_U(v)=sup_{η∈U}vᵀη`.

### Theorem 4: sharp smallest attainable restricted error

For a fixed selected feature set and coefficient budgets as in Theorem 3,

\[
\inf_{\eta\in U}e_b(z;\widehat y+E\eta)
=\max_{\substack{A_0^\top u=0\\\|u\|_1\le1}}
\left[
u^\top\widehat y-h_U(-E^\top u)
-\sum_jb_jz_j|u^\top\phi_j|
\right].
\tag{T6}
\]

In particular, a value greater than `ε` rejects the selected decoder class for **every** true response bank consistent with the uncertainty set. This is a sharp partial-identification calculation for a fixed report and decoder class, rather than an edgewise outer bracket for a chromatic number.

**Proof.** Substitute (T4). The dual domain is compact convex, `U` is compact convex, and the objective is continuous, linear in `η`, and concave in `u` (a linear term minus absolute-value terms). The elementary convex minimax theorem exchanges the minimum over `η` and maximum over `u`. Minimising `uᵀEη` over `U` gives `−h_U(−Eᵀu)`. For a polyhedral `U`, the same equality follows directly from the dual of the joint linear programme in `η,β,γ,t`. ∎

**Uniform independent error box.** If `Y={y:||y−ŷ||∞≤δ}` and the decoder prediction set `M` is any nonempty closed set, then

\[
\inf_{y\in Y}\operatorname{dist}_\infty(y,M)
=\max\{\operatorname{dist}_\infty(\widehat y,M)-\delta,0\}.
\tag{T7}
\]

The reverse triangle inequality proves the lower bound. For the upper bound, take a closest prediction `m` (or a limiting sequence if needed) and move each coordinate of `ŷ` toward `m` by at most `δ`; the residual is the coordinatewise excess over `δ`. In the present affine/box models a closest prediction exists.

By contrast, if **one fixed decoder prediction** must handle every possible response bank, then

\[
\inf_{m\in M}\sup_{y\in Y}\|y-m\|_\infty
=\operatorname{dist}_\infty(\widehat y,M)+\delta.
\tag{T8}
\]

For fixed `m`, the inner supremum is `max_i(|ŷ_i−m_i|+δ)`. Taking the infimum proves (T8). This remains true if `M=R^n`: the robust error is `δ`, not zero. It is incorrect to reuse the nullspace dual for the frozen robust-decoder problem without accounting for cancellation of opposite residual multipliers.

**Shared offset illustration.** In XOR, suppose `y=ŷ+η1`, `|η|≤δ`, and the decoder includes an intercept. Every contrast has `uᵀ1=0`, so the attainable affine error remains exactly `1/2` for every `η`; the uncertainty does not weaken the rejection. If a single fixed intercept must be chosen before the offset is revealed, the robust error is `1/2+δ`. These are different experiments and must have different labels in a paper.

**Transport of a witness.** Let a normalised baseline witness satisfy `Aᵀu=0`, `||u||₁=1`, `uᵀy=L`. On matched cases with perturbed data `A′=A+ΔA`, `y′=y+Δy`, suppose decoder coefficients satisfy `||β||₁≤B`. Then every permitted decoder has error at least

\[
L-\|\Delta y\|_\infty-B\|\Delta A^\top u\|_\infty.
\tag{T9}
\]

This follows by expanding `uᵀ(y′−A′β)` and Hölder's inequality. In particular, coordinatewise feature perturbation at most `δ_A` gives the simpler lower bound `L−δ_y−Bδ_A`. Without a coefficient bound, approximate annihilation does not certify transport: an arbitrarily small feature component aligned with the missing response can be magnified by an arbitrarily large coefficient. This result only transports between matched cases with justified perturbation bounds; it does not turn a finite bank into a population guarantee.

## 5. Finite precision and verification

The unbounded-real affine optimum is a lower-bound oracle for a decoder with quantised coefficients. It is not automatically an upper certificate in that class. If `β*` is a fitted or exact coefficient vector and `β̃` rounds every coefficient to a grid of width `κ`, then

\[
\|y-A\widetilde\beta\|_\infty
\le \|y-A\beta^*\|_\infty
+\frac\kappa2\max_i\sum_j|A_{ij}|.
\tag{T10}
\]

This is the triangle inequality with per-coefficient rounding error at most `κ/2`. A strict residual margin can therefore be converted into an explicit fractional-bit budget. At exact equality with the tolerance, rounding may destroy feasibility. Coefficient magnitude and total encoded coefficient bits should be reported alongside feature count.

For rational `A,y`, an LP dual proposed in floating point can be independently certified without trusting solver status: retain its support, reconstruct rational weights, verify `Aᵀu=0` and `||u||₁=1` exactly, and compute `uᵀy` as a rational number. If a response has known numerical uncertainty, subtract the exact allowance before comparing with `ε`. A sparse exact witness validates the lower bound even when the global minimax optimum has not been proved. A matching rational primal predictor certifies the optimum.

For a binary or ternary intervention design (`A_ij ∈ {−1,0,1}`), these computations are particularly simple. In a one-dimensional nullspace on the extracted support, compute an integer null vector and normalise its ℓ₁ norm. When the support is degenerate, retain only an extreme nullspace witness or solve a small rational linear system; do not assert the `r+1` support bound from a dense approximate solver output.

## 6. What to execute in the trained-system application

1. Freeze the report grammar, threshold precision, admissible contexts, feature costs, decoder formula, tolerance, and split before evaluating locked interventions. Record whether the decoder has an intercept; paired-difference designs commonly do not.
2. Compute both the achieved fitted error and the affine oracle's dual lower bound. Ridge MSE training and L∞ certification answer different questions. A large ridge error alone is not an impossibility certificate.
3. On discovery cases, retain an exact sparse witness for a failed baseline. For each candidate repair feature, publish the signed contrast `uᵀφ_j`; a zero contrast cannot repair that particular witness, and the bounded version quantifies the required capacity.
4. Select a repair using only discovery outcomes; ideally compare contrast-guided, MSE-greedy, and naive equal-cost features. The theorem does not make greedy selection globally optimal.
5. Freeze the repaired encoder and coefficients, quantify their encoded cost, and evaluate locked intervention responses, including an unseen intervention strength. Report failures and any unresolved tolerance boundary. A discovery witness identifies a cause; held-out improvement establishes bounded practical usefulness.
6. If computationally feasible, run exact contrast-cover search on a small frozen candidate library. Report the realised cost-optimal repair or a certified lower/upper gap. This provides a stronger methodological demonstration than another constructed graph example.

## 7. Other routes examined and why they are secondary

**Pairwise predicate repair.** For an unrestricted lookup decoder and frozen predicate library, each incompatible pair must be separated by some selected predicate. This is exactly weighted hitting set on the pairwise discernibility sets. The proof is immediate from the graph criterion. It is useful as a baseline and lower-bound diagnostic but has clear established antecedents in feature reduction and test cover.

**Sharp radius under correlated uncertainty.** For a fixed report, the unrestricted ℓ∞ midpoint radius is half the largest within-fibre coordinate contrast. Under affine shared uncertainty, the smallest attainable radius is an LP and the largest is a maximum of support functions. This improves sharpness over independent edgewise bounds, but it is a direct convex-geometric calculation and does not solve the restricted decoder gap by itself.

**Context or uncertainty universality gap.** With states `v∈{0,1}^m`, common context `θ` in the probability simplex, and scalar response `r_v(θ)=vᵀθ`, every fixed context needs at most two report labels at tolerance `1/4`, since responses lie in `[0,1]`. At simplex vertices every distinct state pair differs by one in at least one context, so a single context-independent report needs all `2^m` labels. This sharp exponential gap is useful pedagogically but is an indexing/one-way communication phenomenon, not a persuasive standalone novelty claim.

**Generic sequential composition.** Lipschitz propagation of per-stage errors is straightforward, and without support closure even individually good local audits need not compose. A compositional extension would require an actual model and intervention closure contract; absent that, it would enlarge the paper more than strengthen it.

## 8. Claims that are and are not justified

Theorems 1–4 and the examples are fully proved here. Their central mathematical tools are classical. The proposed method detects failure of a declared restricted decoder and can identify a minimum-cost repair within a finite frozen library. It does not establish semantic truth, causal-graph identification, uniqueness of explanation, universal coverage, or superiority of native/group/dictionary representations.

The trained-system application is being developed separately; no empirical results are asserted in this memo. A completed, consequential, independently checked application could strengthen the significance claim. The theory provides exact audit evidence and an algorithmic repair formulation; it cannot substitute for that evidence or certify scientific significance.

## 9. Sharpening the fixed-report uncertainty calculation

For completeness, here is the precise useful replacement for an edgewise uncertainty bracket when the **report itself is fixed**. Let `I_t` be its fibres, and let the scalar response in coordinate `k` be `r_ik(η)=a_ik+B_ik η`, with the same `η∈U` for all entries. Define directed within-fibre contrasts

\[
\mathcal D=\{(i,j,k):i,j\in I_t\text{ for some }t\},
\qquad
d_{ijk}(\eta)=a_{ik}-a_{jk}+(B_{ik}-B_{jk})\eta.
\]

The optimal unrestricted midpoint decoder's error at that particular response bank is

\[
\rho_T(\eta)=\tfrac12\max_{(i,j,k)\in\mathcal D}d_{ijk}(\eta).
\tag{T11}
\]

The self-pairs make the maximum nonnegative. If `U` is a polytope, the **smallest attainable error** is the LP minimising `t` over `η∈U` subject to `2t≥d_ijk(η)` for all contrasts. The **largest attainable error, with the decoder allowed to adapt to the true bank**, is

\[
\sup_{\eta\in U}\rho_T(\eta)
=\tfrac12\max_{(i,j,k)\in\mathcal D}
\left[a_{ik}-a_{jk}+h_U((B_{ik}-B_{jk})^\top)\right].
\tag{T12}
\]

**Proof.** In each fibre and coordinate, a common scalar prediction has minimax error half the response range, achieved by its midpoint. The maximum over fibres and coordinates gives (T11). The minimum is an epigraph LP. The maximum commutes with the finite contrast maximum, giving (T12). ∎

The **single frozen decoder** has a different minimax error. Write `U_ik=sup_η r_ik(η)` and `L_ik=inf_η r_ik(η)`. Its optimum is

\[
\rho_T^{\rm fixed}=
\tfrac12\max_{t,k}
\left[\max_{i\in I_t}U_{ik}-\min_{j\in I_t}L_{jk}\right].
\tag{T13}
\]

The same midpoint argument now covers the union of all possible scalar responses. It permits different nuisance realisations at the upper and lower extremes because one fixed prediction must handle both. Thus correlated uncertainty can disappear from a contrast in (T12) and still matter in (T13). Neither formula alone computes the sharp maximum chromatic number when the encoder is also allowed to change with the response bank.

For the supplied review's three-state example `ŷ=(0,11/10,11/5)`, `δ=1/2`, and `ε=1`, the exact attainable label counts are `{1,2}`. One label is attained at `(1/2,11/10,17/10)`, whose span is `6/5≤2`. Two labels are necessary at `(-1/2,3/5,27/10)` because its endpoints differ by `16/5>2`, and two always suffice: three ordered scalar points all separated pairwise by more than two would require a total span greater than four, whereas the permitted span is at most `16/5`. The edgewise construction yields `[1,3]` and is therefore only an outer bracket. This example and diagnosis belong to the supplied review and should be attributed accordingly.

## 10. Executed implementation-control receipt

`check_theory.py` was run successfully with one process and one BLAS thread. It performs six exact rational XOR coefficient-budget cases, 64 separate primal/dual comparisons (including degenerate designs), 51 rationally reconstructed exact nullspace witnesses, 32 bounded-coefficient primal/dual comparisons, four synergy subsets, 56 vertex-cover reduction subsets, and 12 contrast-cover searches checked against exhaustive feature-subset enumeration. Those searches generated 126 exactly verified violated contrast cuts. All passed. The largest retained witness used seven cases.

The receipt is `checks.json`, with random seed `20261015`, NumPy `2.4.6`, SciPy `1.17.1`, and SymPy `1.14.0`. The controls establish implementation agreement on these constructions; they do not establish novelty, empirical prevalence, or an independent proof of the general theorems.
