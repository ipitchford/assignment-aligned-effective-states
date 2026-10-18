# Prospective causal-state sufficiency: a separate contract and exact certificates

This is a retrospective, weight-only supplementary analysis of the three already trained shallow models. No sampled training or test observations, empirical prediction outcomes, or intervention outcomes were read to select the constructions below. It does not change any frozen experiment. Its computational claims are in `results.json`; `verify.py` independently checks the principal certificates using integer determinants and rational arithmetic, without an optimiser. All indices are zero based.

The main substantive result is a separation between two questions. A compact report of the current output can support accurate reconstruction of a change when reports of **both endpoints** are supplied. The same report can fail to predict the response to a known action from the **initial state alone**, even if the current output is reported at infinite precision. For each of the three trained models we construct two exact, valid continuous input vectors with identical current output vectors but a group-ablation response gap exceeding nine centered-logit units. This is an obstruction for every decoder of the current-output-only report, rather than evidence about one fitted decoder.

The underlying linear-state characterisation is classical observability and exact quotient theory. It should be credited as an application, not claimed as a new theorem about representation learning. The contribution available here is a concrete audit contract, intervention-dependent cost, realised counterexamples, and reproducible checks.

## 1. The prospective contract

Let the internal state be $h\in\mathbb R^n$, the declared output be $y(h)=Ch+c_0\in\mathbb R^q$, and each allowed action $a\in\mathcal A$ act by $h\mapsto D_a h$. The fixed bias $c_0$ is known to the decoder and cancels in response differences. For a word $w=(a_1,\ldots,a_t)$, let $D_w=D_{a_t}\cdots D_{a_1}$, including $D_\varnothing=I$.

There are three different contracts.

1. **Retrospective paired response:** receive $T(h)$, $T(D_a h)$, and possibly the action, then reconstruct $C(D_a-I)h$.
2. **Prospective response:** receive $T(h)$ and the action or action word, then reconstruct $C(D_w-I)h$. An exact one-step version demands this only for words of length one.
3. **Prospective state quotient:** in addition to an output decoder, demand report transition maps so that the updated report can be computed exactly from the previous report and action: $T(D_a h)=F_a(T(h))$.

The third implies prediction of arbitrarily long action words. The second does not in general supply an exact transition map for the encoder itself. A decoder that carries a continuous approximate state forward can satisfy a uniform output-error contract without producing the code that re-encoding the true state would have produced.

The state domain is part of the contract. Results on all of $\mathbb R^n$, on a hidden-state box, on continuously reachable network inputs, and on a finite image bank have different meanings. In particular, a finite bank can admit an injective low-dimensional report and unrestricted memorisation even when no comparable report works on an open state domain.

## 2. Classical exact linear quotient characterisation

**Proposition 1 (linear observability quotient, stated for the audit).** Let $R\in\mathbb R^{d\times n}$ have full row rank. The following are equivalent:

\[
\exists O,\{K_a\}_{a\in\mathcal A}:\quad C=OR,\qquad RD_a=K_aR\quad(a\in\mathcal A);
\tag{1}
\]

\[
\ker R\subseteq\ker C,\qquad D_a\ker R\subseteq\ker R\quad(a\in\mathcal A).
\tag{2}
\]

Define

\[
V=\operatorname{span}\{\operatorname{rows}(CD_w):w\in\mathcal A^*\}.
\tag{3}
\]

The smallest possible dimension of a report satisfying (1) is $\dim V$. Every minimal report has row space $V$. Taking any basis of $V$ as the rows of $R$ gives an output map and exact transition maps. The iteration

\[
V_0=\operatorname{row}(C),\qquad V_{t+1}=V_t+\sum_{a\in\mathcal A}V_tD_a
\tag{4}
\]

stops after at most $n-\operatorname{rank}C$ strict dimension increases; its terminal space is $V$.

**Proof.** If (1) holds and $Rv=0$, then $Cv=ORv=0$, and $RD_av=K_aRv=0$. This proves (2). Conversely, a linear map factors through the surjection $h\mapsto Rh$ precisely when it vanishes on $\ker R$. Apply this fact to $C$ and to each $RD_a$ to obtain (1).

If (1) holds, $C=OR$, and induction on word length gives $CD_w=OK_wR$. Thus $V\subseteq\operatorname{row}(R)$, proving the lower bound. Conversely, $V$ contains the rows of $C$ and is invariant under right multiplication by every $D_a$. A basis matrix $R$ therefore satisfies (1). Equality of dimensions forces the same row space. Finally, the increasing chain (4) enumerates rows of words of bounded length. If one stage does not increase, it is invariant and all later stages agree. Strict increases cannot continue beyond dimension $n$. ∎

The interpretation is that the report may discard only differences that remain invisible under every future permitted action. In control terminology this is the unobservable subspace and its quotient. General switched-system realisation and minimality precede this audit; see [Petreczky, Bako and van Schuppen, *Realization theory of discrete-time linear switched systems*](https://arxiv.org/abs/1103.1343). The exact observation-preserving quotient viewpoint is also established in [Pappas, *Bisimilar linear systems*, Automatica 39 (2003), 2035–2047](https://www.georgejpappas.org/wp-content/uploads/2024/04/AUT03.pdf).

For an exact **one-step** linear output decoder, the minimum row space is instead

\[
\operatorname{row}\begin{bmatrix}C\\ CD_a:a\in\mathcal A\end{bmatrix}.
\tag{5}
\]

This can be smaller than (3). For example, take $C=(1,0,0)$ and a single action with matrix $D=\begin{pmatrix}0&1&0\\0&0&1\\0&0&0\end{pmatrix}$. The present output and one-step output require two coordinates, while the two-step output reveals the third. Thus a claim about one-step responses should not silently be promoted to a compositional state claim.

These statements also hold if the identities are required only on a set spanning $\mathbb R^n$, **provided the output and transition maps are linear as in (1)**. Equality of two linear maps on a spanning set implies equality everywhere. The spanning-set argument does not justify a lower bound against an unrestricted nonlinear lookup decoder on a finite sample. On a domain containing an open ball, however, an arbitrary decoder of a linear report must still respect local report kernels; this yields the same exact factorisation necessity for linear target maps.

## 3. Diagonal actions make intervention granularity explicit

**Proposition 2 (action-signature decomposition).** Suppose all $D_a$ are diagonal, and the action set is finite. Coordinates $j,k$ have the same action signature when $(D_a)_{jj}=(D_a)_{kk}$ for every $a$. Let $G_1,\ldots,G_m$ be the resulting equivalence classes and $P_\ell$ their coordinate projectors. Then

\[
V=\bigoplus_{\ell=1}^{m}\operatorname{row}(CP_\ell),\qquad
\dim V=\sum_{\ell=1}^{m}\operatorname{rank}(C_{:,G_\ell}).
\tag{6}
\]

**Proof.** Every product of the diagonal actions is constant on each signature class. Therefore the unital algebra they generate is contained in the span of $P_1,\ldots,P_m$. For the reverse inclusion, fix signature $s_\ell$. For each other signature $s_r$, choose an action coordinate $a(r)$ on which they differ. The polynomial

\[
p_\ell(z)=\prod_{r\ne\ell}
\frac{z_{a(r)}-(s_r)_{a(r)}}{(s_\ell)_{a(r)}-(s_r)_{a(r)}}
\]

equals one at $s_\ell$ and zero at every other signature. Evaluation on the commuting diagonal matrices gives $p_\ell(D_a)=P_\ell$. Consequently the generated algebra is exactly the span of these projectors, and multiplication by $C$ gives the first equality in (6). The summands have disjoint coordinate supports, so the sum is direct and its dimension is the sum of their ranks. ∎

For disjoint coordinate groups and actions $D_g=I-\alpha_gP_g$, with every $\alpha_g\ne0$, the proof is even shorter: $CP_g=(C-CD_g)/\alpha_g$. Products of the group projectors add no rows because distinct projectors multiply to zero and each is idempotent. Thus present output plus the one-step group responses already contain the entire action closure. This is a special case of elementary simultaneous diagonalisation and classical observability, not a proposed new general theorem.

The useful research implication is that the necessary state dimension depends on which distinctions the allowed interventions can expose. A single scalar attenuation of the whole layer preserves the original output row space. Independently attenuating finer groups can reveal directions previously cancelled in that output. This dependence is a property of the intervention family and downstream computation; it does not establish that the coordinates are semantic concepts.

Even commuting attenuation masks can make compositions reveal information that every individual action hides. For a four-coordinate example let
\[
C=(1,1,1,1),\quad
D_a=\operatorname{diag}(1,1,0,0),\quad
D_b=\operatorname{diag}(1,0,1,0).
\]
The bank of present and individual-action outputs has rank three, while its action closure has rank four because the product adds $CD_aD_b=(1,0,0,0)$. The states $h_+=(1,0,0,1)^T$ and $h_-=(0,1,1,0)^T$ have identical present output two and identical outputs one after either individual action, but composition outputs one and zero. Thus even an exact, unrestricted decoder of the entire single-action output bank cannot resolve the composed query. This is a concrete compositional counterexample, not a claim about the present four disjoint groups, for which the one-step bank already closes.

The same example shows that the exact dimension required by an action family need not be a submodular function of the allowed actions. With neither action, the dimension is one; with either action alone it is two; with both it is four. The marginal cost of adding one action can therefore increase after another action has been admitted. A greedy action-budget procedure cannot assume diminishing returns merely because it builds an observability space. This is a simple consequence of the classical word closure, rather than evidence of a new general approximation theorem.

## 4. Exact rank results on the trained shallow models

Each model has 32 ReLU hidden units and ten centered outputs. The dyadic exported head has stored integers

\[
\mathsf C_{jk}=10(\mathsf W_2)_{jk}-\sum_{l=1}^{10}(\mathsf W_2)_{jl}.
\]

For the hidden integer-scale state $s=65536h$, the bias-free centered output is $\mathsf C^Ts/D$, with $D=65536\cdot10\cdot4096=2684354560$. The fixed centered output bias does not affect ranks or response differences. In the notation of the propositions, the matrix $C$ is the corresponding normalized $10\times32$ head, rather than the transposed stored array $\mathsf C$.

The four fixed intervention groups partition the 32 coordinates into blocks of eight. All three models have the following exact ranks.

| Seed | Current centered head | Each of four group blocks | Four-group one-step bank | Action closure | Two coarsened groups |
|---|---:|---:|---:|---:|---:|
| 17 | 9 | 8, 8, 8, 8 | 32 | 32 | 18 |
| 29 | 9 | 8, 8, 8, 8 | 32 | 32 | 18 |
| 43 | 9 | 8, 8, 8, 8 | 32 | 32 | 18 |

“Two coarsened groups” combines the first two original groups and the last two. Each of these 16-coordinate head blocks has rank nine. If only whole-layer scalar attenuation were allowed, the action-closed rank would be nine. Starting with the present output and adding original groups in their frozen order gives ranks $9,17,25,32,32$. These are exact structural results, not numerical thresholds applied to singular values.

The construction script finds nonzero minors modulo the prime 1000003. A nonzero modular determinant certifies a nonzero integer determinant. The head has upper rank nine because its ten rows sum to zero; each eight-coordinate block has upper rank eight; the full bank has upper rank 32. The displayed minor lower bounds reach these algebraic upper bounds. The separate verifier evaluates the selected minors with fraction-free integer Bareiss elimination and checks the modular receipts against the exact determinants.

There is also a domain check for linear decoders: 98 deterministic input probes in $[0,16]^{64}$ have hidden states spanning all 32 dimensions. These probes are the zero input, the all-16 input, the 64 single-coordinate 16-valued inputs, and 32 corners chosen to maximise the first-layer preactivation of each unit. A nonzero $32\times32$ hidden-state minor verifies the span. The states also **affinely** span all 32 dimensions: subtracting the hidden state of the first probe from every other probe leaves a matrix with a separately certified nonzero $32\times32$ minor. This additionally rules out saving report dimensions by allowing affine decoder offsets: differences eliminate the offsets and force the same matrix factorisation. Therefore the rank-32 necessity for linear reports with affine output/transition factorisations is not solely an artefact of allowing unreachable hidden states. It holds already when those identities are demanded on these continuously admissible input probes. These probes are constructed points, not a natural-image distribution.

## 5. Realised output aliases: a decoder-independent prospective failure

**Proposition 3 (same report, different action response).** Suppose $T(h_+)=T(h_-)$, the action $a$ is the same, and a target coordinate differs by

\[
\Delta=|[C(D_a-I)h_+]_k-[C(D_a-I)h_-]_k|.
\]

Every prospective decoder based only on $T(h)$ and $a$ has maximum absolute error at least $\Delta/2$ on these two states. The same holds for the two post-action outputs whenever their initial outputs agree.

**Proof.** The decoder receives identical information and returns one value $v$. The triangle inequality gives $\Delta\le|r_+-v|+|r_--v|\le2\max(|r_+-v|,|r_--v|)$. If initial outputs agree, their difference cancels in comparing post-action outputs. ∎

We construct $h_+,h_-$ by two actual network inputs $x_+,x_-\in[0,16]^{64}$. At the input centre $x_0=(8,\ldots,8)$, every first-layer preactivation is nonzero. We do **not** assume that all hidden units are active: the three masks have 27, 30 and 26 active units respectively. Let $A$ be the diagonal active-unit mask. The Jacobian of the centered output within that activation cell is proportional to

\[
E=\mathsf C^T A\mathsf W_1^T.
\]

This matrix has rank nine. A direction $v$ with $Ev=0$, $8\pm v_i\in[0,16]$, and

\[
|(\mathsf W_1^Tv)_j|\le
|8\sum_i(\mathsf W_1)_{ij}+16(\mathsf b_1)_j|
\tag{7}
\]

keeps both endpoints in the same closed ReLU cell and makes their centered outputs identical. Within this polytope a numerical LP proposes directions maximising each of the 40 group/output contrasts. The chosen direction is then projected onto the nullspace **exactly** by solving an invertible rational $9\times9$ subsystem, and rescaled rationally to satisfy all bounds. The final certificate is the pair of rational input vectors; its validity is independent of the LP status or objective accuracy.

Exact reevaluation of the full first layer and head gives:

| Seed | Group | Output coordinate | Full-ablation response gap | Forced error, full ablation | Forced error, half ablation |
|---|---:|---:|---:|---:|---:|
| 17 | 3 | 1 | 11.0195414564 | 5.5097707282 | 2.7548853641 |
| 29 | 0 | 2 | 10.2827756116 | 5.1413878058 | 2.5706939029 |
| 43 | 2 | 4 | 9.1795789430 | 4.5897894715 | 2.2948947357 |

Here full ablation is the fixed strength one, so the selected group is multiplied by zero. Half ablation is the fixed strength one half, so that group is multiplied by one half. The response is the post-action centered logit minus its pre-action value. The table gives the absolute difference between the two states' responses to that same action.

The decimal table is for reading; the JSON records contain exact fractions for the inputs, hidden states, common outputs and gaps. The independent verifier checks every input coordinate bound, recomputes the first-layer preactivations and ReLU outputs, checks equality of all ten current centered projections, and checks the action gap. It explicitly adds the centered output bias
\[
(c_0)_k=\frac{10(\mathsf b_2)_k-\sum_l(\mathsf b_2)_l}{10\cdot4096}
\]
to both projections and checks equality of the complete current centered-logit vectors. The biases are fixed across the pair; they are included in the certificate and cancel in response differences. The verifier also computes the existing 77-bit head-projection codes and confirms that the pair has identical codes.

In fact the lower bound applies to **every** encoder of the form $T(h)=f(Ch+c_0)$, regardless of its bit budget or nonlinear decoder. Supplying the complete exact current output still cannot distinguish these pairs. The obstruction is therefore not an artefact of projection quantisation, ridge fitting, or collision frequency in a finite test sample. It is an exact failure on constructed continuous inputs in the declared input range. It does not establish occurrence among natural digit images, a distributional failure rate, or any claim about human semantic features.

## 6. Bounded-error alternatives and their scope

Exact rank is a zero-error statement. Rank 32 does not imply that 32 real-valued coordinates, any particular bit cost, or lossless state recovery is necessary at tolerance one. Error and admissible domain must be analysed separately.

**Proposition 4 (exact worst-case recovery error for a fixed linear report on a box).** Let $H=m+B[-1,1]^n$, where $B$ is an invertible diagonal scale matrix. Fix a linear report $Rh$ and linear target bank $Lh\in\mathbb R^q$. Allow an arbitrary decoder of the report and use $\ell_\infty$ output error. Then

\[
\inf_g\sup_{h\in H}\|Lh-g(Rh)\|_\infty
=\max_{\substack{\|z\|_\infty\le1\\RBz=0}}\|LBz\|_\infty.
\tag{8}
\]

**Proof.** If $h_+,h_-\in H$ have the same report, write $h_+-h_-=2Bz$. Then $\|z\|_\infty\le1$ and $RBz=0$. Thus the coordinate width of every target fibre is at most twice the right side. Choosing each coordinate's interval midpoint in each fibre attains at most the right side. Conversely, for any feasible $z$, the two states $m+Bz,m-Bz$ lie in $H$, share a report, and have target separation $2LBz$. Proposition 3 supplies the lower bound. Compactness supplies maxima. ∎

This is an elementary instance of optimal recovery/fibre-radius reasoning. It is included to make the approximate audit operational, not asserted as a new width theorem. The right side is the maximum of $2q$ linear programs. It also shows why a rank-only calculation can be practically misleading: small newly visible directions may have small amplitude on the declared box. Approximate observation-preserving dynamics has a substantial prior literature, including [Girard and Pappas, *Approximate bisimulation relations for constrained linear systems*, Automatica 43 (2007), 1307–1317](https://doi.org/10.1016/j.automatica.2007.01.019); the error contract here should not be presented as the invention of approximate bisimulation.

For a simple explicit sufficient repair, quantise every hidden coordinate $h_j\in[0,H_j]$ into $2^{b_j}$ equal intervals and let $\tilde h_j$ be the interval midpoint. Then $|h_j-\tilde h_j|\le H_j2^{-b_j-1}$. For any diagonal contraction word $D=\operatorname{diag}(d_j)$, $0\le d_j\le1$, predict the output and response by $CD\tilde h+c_0$ and $C(D-I)\tilde h$. Both errors obey

\[
\max_k\frac12\sum_j |C_{kj}|H_j2^{-b_j}.
\tag{9}
\]

Indeed $|d_j|\le1$ and $|d_j-1|\le1$, so the triangle inequality gives (9). Products of the permitted attenuation actions remain diagonal contractions. The guarantee therefore holds for arbitrary compositions from the same initial report, without accumulating an error per action, provided the decoder tracks the composed contraction or evolves the original midpoint estimate without re-quantising at every step.

The existing weight-only sufficient-bound allocation receipts at bound two immediately give the following prospective tolerance-one repairs:

| Seed | Bits for one initial report | Exact maximum bound |
|---|---:|---:|
| 17 | 187 | 42752472483/42949672960 |
| 29 | 198 | 4253432089/4294967296 |
| 43 | 181 | 42616511823/42949672960 |

The verifier recomputes all ten rational bounds and stores the actual maximum fraction in `verification.json`. These costs are optimal only for the declared sum-of-absolute-gains sufficient-bound integer allocation problem established in the separate precision work. They are **not** proved optimal among all prospective encoders, or even for the restricted four-group action family. A single-group query permits sharper group-specific sums, and additional structure can make (9) conservative.

These midpoint estimates provide bounded output prediction, not automatically an exact transition of the finite code. There is a useful special-case distinction. For powers of two, choose the right-closed quantiser

\[
Q_b(x)=\max\{0,\lceil2^b x/H\rceil-1\},\qquad0\le x\le H.
\tag{10}
\]

Then $Q_b(x/2)=\lfloor Q_b(x)/2\rfloor$ and $Q_b(0)=0$. To verify the first identity, for $x>0$ put $q=Q_b(x)$, so $q<2^b x/H\le q+1$. Dividing this interval by two gives code $\lfloor q/2\rfloor$; the zero endpoint is direct. The product code therefore has exact report transitions for coordinate halving and zeroing and retains the midpoint error bound. This is an alternative mathematical design, not a modification to the frozen left-closed implementation. Multiplication by $3/4$ can split a uniform bin across two bins, so the exact transition identity should not be extended to arbitrary strengths.

There is also a precise limit to demanding finite exact state transitions for every attenuation strength. Suppose $Q:[0,1]\to\mathcal Q$ has finite range, $Q(rx)=F_r(Q(x))$ for every $r\in[0,1]$, and an output decoder $g$ satisfies $|g(Q(x))-x|\le\varepsilon<1/2$ for all $x$. Finiteness yields two distinct $2\varepsilon<x<y\le1$ with $Q(x)=Q(y)$. Put $r=x/y$. The transition property implies $Q(r^ny)=Q(y)$ for every $n\ge0$. The common decoded value is within $\varepsilon$ of $y$ and of numbers tending to zero; hence $y\le2\varepsilon$, a contradiction. Thus finite approximate output codes and exact code-transition closure cannot be demanded simultaneously at arbitrary precision for the continuum of all attenuation strengths. This elementary example does not rule out finite action alphabets, approximate transitions, or the prospective bounded-error predictor in (9).

## 7. What this adds, and what it does not

The prospective contract materially repairs the explanatory interpretation of the earlier paired audit. It prevents a report that simply recomputes the requested output at both endpoints from being mistaken for a sufficient causal state. The exact realised aliases show the distinction in all three trained shallow models and at the declared tolerance; the action-closure calculation explains which information is missing and why changing intervention granularity changes the required exact linear state.

The linear quotient theorem, invariant-kernel reasoning, rank closure, diagonal signature decomposition, and approximate state abstraction all have established mathematical foundations. Their translation here does not by itself justify a high originality rating. Nor does full observability imply meaningful neuron labels: full rank can arise generically from independently manipulating sufficiently small groups. A target bank that computes every requested future response inside the encoder remains a possible task-specific report, with its computation and stored coefficients charged explicitly. A credible paper should present these facts as controls on the meaning of its sufficiency claim.

The constructed aliases are continuously admissible network inputs, but they were deliberately selected from weights and are not natural-digit pairs. The rank certificate addresses exact linear reports and transition maps; it is not a lower bound on arbitrary finite or nonlinear encoders over a finite sample. The explicit precision repair proves an upper bound for a stated box and action class; it is not a semantic representation or a universal optimum. The more consequential empirical test is whether coarse reports fail to support correct prospective action choices on naturally occurring states, under a frozen action and target definition. That empirical question is being studied separately and is not answered by these receipts.

Accordingly this work is a meaningful change in research question and an exact explanatory counterexample, with conventional control-theoretic machinery. It strengthens the manuscript beyond a retrospective reconstruction audit; significance still depends on the integrated empirical result, decision relevance, breadth, reproducibility, and careful comparison with existing interpretability and abstraction methods. It is not established by these lemmas or these three small models alone.

## 8. Replay

Construction: run `audit.py` with NumPy, SciPy and SymPy. Verification: run `verify.py` with NumPy; the verifier uses no optimiser or symbolic algebra. Both use only exported weights, declared action groups, earlier weight-only precision allocations and the target-projection receipt. They do not load image data, discovery response banks, or held-out outcomes.

`results.json` contains source hashes, exact rank minors and rational realised aliases. `verification.json` contains its own source hashes, full integer minor determinants, identical 77-bit alias codes and exact prospective precision bounds. Independent replay establishes the scope of those computations; it is not independent scientific replication, peer review, a formal proof assistant check, or a novelty assessment.
