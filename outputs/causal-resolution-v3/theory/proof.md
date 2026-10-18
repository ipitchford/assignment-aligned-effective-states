# Exact bit allocation for the certified native precision repair

15 October 2026. This is **post-test supplementary optimisation of a weight-only design**. It did not load test outcomes, revise the frozen precision reports, or evaluate a new locked test. The only model-dependent inputs were the already frozen `M`, `C`, and incumbent bit allocations in each `precision-gain-ε.npz` file. Input hashes are retained in the receipts.

## Result and its scope

At paired-response tolerance `ε=1`, the exact minimum numbers of bits per state satisfying the declared global box **sufficient bound** are:

| Model seed | Uniform allocation | Earlier greedy allocation | Exact optimum | Exact maximum certified error of the retained optimum |
|---|---:|---:|---:|---:|
| 17 | 256 | 220 | **218** | `42833232791/42949672960 ≈ 0.997289` |
| 29 | 256 | 230 | **230** | `42806650647/42949672960 ≈ 0.996670` |
| 43 | 256 | 212 | **211** | `8531403393/8589934592 ≈ 0.993186` |

The improvements over greedy are two, zero, and one bits per state. The savings relative to 256 uniform bits are 38, 26, and 45 bits. A paired report uses twice the state-bit count. The bit count does not include the fixed dense readout or the quantiser specification; those are still part of an explanation's total cost ledger.

These are exact optima **within the separable per-unit bit allocation problem using this sufficient bound**. They are not lower bounds on every encoder, every native report, the actual worst-case model error, or semantic explanation complexity. The bound takes absolute readout gains coordinatewise and can be conservative because it discards cancellation and coupling in reachable hidden states.

The primary result receipt is `results/precision_optimality.json`; `results/precision_verification.json` records a separate exact replay that calls no optimiser. `precision_optimality.py` performs candidate search and verification. Every reported primary optimum has an exact primal allocation and an exact lower certificate. Floating-point solver status is not used as the mathematical proof.

## Declared optimisation problem

Let `j=1,…,32` index hidden units and `k=1,…,10` index centred output coordinates. The precision contract supplies nonnegative integer hidden-box bounds `M_j` and centred integer readout coefficients

\[
C_{jk}=10W^{\mathrm{int}}_{2,jk}-\sum_{\ell=1}^{10}W^{\mathrm{int}}_{2,j\ell}.
\]

Define the exact nonnegative gains

\[
a_{jk}=\frac{M_j|C_{jk}|}{D},\qquad
D=65536\cdot10\cdot4096=2684354560.
\]

The integer bit allocation `b_j≥0` gives `2^{b_j}` uniform bins on the certified hidden interval, decoded to bin midpoints. The difference of two endpoint reconstruction errors is at most one full bin width. Consequently the fixed linear readout's paired error in coordinate `k` is at most

\[
E_k(b)=\sum_{j=1}^{32}a_{jk}2^{-b_j}.
\]

The problem solved here is

\[
C^*(\varepsilon)=\min_{b\in\mathbb Z_{\ge0}^{32}}
\left\{\sum_jb_j:E_k(b)\le\varepsilon\quad(k=1,\ldots,10)\right\}.
\tag{P}
\]

Zero bits means one constant midpoint bin. Its code requires no state bits, although its approximation contribution remains in the bound. All ten output constraints are enforced with rational arithmetic.

The earlier precision contract supplies the hidden-domain assumptions behind `M_j`: inputs in `[0,16]^{64}`, the exported ReLU model, and coordinate attenuations that keep both endpoint states inside the certified box. This optimisation changes only integer precision allocation. It does not establish the contract's domain outside those assumptions.

## Rational Lagrange lower bound

For any nonnegative multiplier vector `μ∈Q_{≥0}^{10}`, put

\[
s_j=\sum_k\mu_ka_{jk},\qquad
m_j=\min_{b\in\mathbb Z_{\ge0}}(b+s_j2^{-b}),
\]

and

\[
L(\mu)=\sum_jm_j-\varepsilon\sum_k\mu_k.
\tag{1}
\]

Then

\[
C^*(\varepsilon)\ge\lceil L(\mu)\rceil.
\tag{2}
\]

**Proof.** For every feasible allocation,

\[
\begin{aligned}
\sum_jb_j
&\ge \sum_jb_j+\sum_k\mu_k\left(E_k(b)-\varepsilon\right)\\
&=\sum_j(b_j+s_j2^{-b_j})-\varepsilon\sum_k\mu_k\\
&\ge L(\mu).
\end{aligned}
\]

The objective is integer, giving (2). No assumption that `μ` is dual-optimal is required. ∎

Each `m_j` is computed over **all** nonnegative integers; there is no unproved maximum-bit cutoff. For `f_j(b)=b+s_j2^{-b}`,

\[
f_j(b+1)-f_j(b)=1-s_j2^{-b-1}.
\]

The function decreases until this difference becomes nonnegative and increases thereafter. A smallest minimiser is the first `b≥0` satisfying `s_j≤2^{b+1}`. For `b>0`, it also satisfies `s_j≥2^b`. These two integer/rational comparisons certify the global minimum, including equality and zero-gain cases.

The floating-point LP proposes multipliers. They are rounded to nonnegative integers over the fixed denominator `10^9`, and (1) is recomputed entirely as fractions. Their exact numerator vectors and every `m_j` are recorded in the JSON receipts, allowing an independent checker to ignore the LP completely.

At tolerance one:

| Seed | Exact rational `L(μ)` | Approximation | Rounded integer lower bound |
|---|---|---:|---:|
| 17 | `9342247786947828330227/42949672960000000000` | 217.5161565409002 | 218 |
| 29 | `9824022476083989677277/42949672960000000000` | 228.73334763767173 | 229 |
| 43 | `4515480379884831721577/21474836480000000000` | 210.26844065100101 | 211 |

The exact feasible allocations for seeds 17 and 43 match these rounded lower bounds, proving optimality immediately. Seed 29 requires the additional exact exclusion below; claiming a 230-bit proof from `ceil(L)=229` alone would be incorrect.

## Safe finite truncation and exact exclusion

Define nonnegative reduced penalties

\[
\Delta_j(b)=b+s_j2^{-b}-m_j.
\]

The preceding proof gives, for every feasible allocation of cost at most integer `B`,

\[
\sum_j\Delta_j(b_j)\le B-L(\mu).
\tag{3}
\]

Since `f_j(b)≥b`, every such allocation must satisfy

\[
0\le b_j\le\left\lfloor B-L(\mu)+m_j\right\rfloor,
\quad
\Delta_j(b_j)\le B-L(\mu).
\tag{4}
\]

These restrictions produce finite permitted sets `S_j(B)` without excluding any feasible allocation of cost at most `B`. They justify both the small multiple-choice MILP and the exact search. The MILP has one binary indicator for each permitted `(j,b)` and enforces exactly one bit choice per unit, all ten response constraints, and the sum-bit objective. Its role is to find a candidate, whose feasibility is then checked exactly.

For seed 29, set `B=229`. Its exact allowed sets have one or two values per unit; 23 units remain binary choices. Direct product enumeration would have `2^23` candidates. An exact branch-and-bound search rejects the entire set in **581 nodes** using (3), the cost budget, and the following safe per-output bound.

For each remaining suffix of units and output coordinate `k`, dynamic programming computes the minimum possible residual response contribution using at most `c` additional bits:

\[
Q_{i,k}(c)=\min_{b\in S_i(B),\,b\le c}
\{a_{ik}2^{-b}+Q_{i+1,k}(c-b)\},
\qquad Q_{\mathrm{end},k}(c)=0.
\tag{5}
\]

An impossible remaining cost is assigned infinity. Different output coordinates may use different minimising assignments in (5); this relaxes the common-assignment problem and hence still gives valid coordinatewise lower bounds. A branch is rejected when its selected response contribution plus any `Q_{i,k}(remaining bits)` exceeds `ε`, or its reduced penalty exceeds (3). Every retained branch chooses each still-permitted bit value, so a completed search without a feasible leaf proves infeasibility of cost at most `B`.

All comparisons use integers after clearing denominators. At `ε=1`, the response denominator is `687194767360`; the reduced-penalty denominator is `42949672960000000000`, with exact penalty allowance `11452631756010322723`. Of the 581 visited nodes, 113 were rejected by the reduced-penalty bound and 178 by response/cost bounds. No feasible leaf remained. The independently replayed search has the same node count and allowed sets. Therefore seed 29 requires at least 230 bits, matching its exactly feasible 230-bit allocation.

This proof uses classical Lagrangian relaxation, multiple-choice integer optimisation, and exact finite search. The contribution is the certified optimum for the declared trained-model precision audit, not a new general optimisation principle.

## Tolerance sensitivity

The same weight-only calculation was cheap enough to complete all five already declared tolerance settings. Every table entry is an exactly certified optimum of (P).

| Paired tolerance `ε` | Seed 17 | Seed 29 | Seed 43 |
|---:|---:|---:|---:|
| 1/4 | 282 | 294 | 271 |
| 1/2 | 250 | 262 | 241 |
| 1 | 218 | 230 | 211 |
| 2 | 187 | 198 | 181 |
| 4 | 156 | 166 | 151 |

Seed 29 uses exact finite exclusion at all five settings; seed 17 also needs it at `ε=4`. All other entries follow from rounding the rational Lagrange bound. The complete receipts are in `results/precision_optimality_sensitivity.json`. This sensitivity table does not use observed test errors to choose an allocation or tolerance.

## Reproduction and verification

Candidate optimisation and exact arithmetic checks:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python precision_optimality.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python precision_optimality.py --sensitivity
```

To check an existing receipt **without calling LP or MILP**:

```text
python precision_optimality.py --verify results/precision_optimality.json
python precision_optimality.py --verify results/precision_optimality_sensitivity.json
```

The actual interpreter used was `/Users/admin/.venvs/gaia-tess-microlensing/bin/python`. Each optimisation run completed in a few seconds with one process and single-thread BLAS settings. Verification rechecks input hashes, every exact response bound, the rational multipliers and global per-unit minima, and any exhaustive integer exclusion. All primary and sensitivity records passed.

No test results were loaded during this work. The retained allocations have a global mathematical certificate under the frozen model/domain assumptions, but no new empirical held-out result is claimed for these post-test allocations. The frozen pre-test gain allocations remain intact.
