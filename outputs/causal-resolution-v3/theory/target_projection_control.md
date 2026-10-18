# A target-aware output-projection control

15 October 2026. This is a **retrospective, weight-only calibration**, not a revision of the frozen experiment. The calculation loads the certified hidden box and known centred readout weights. It does not load test outcomes, fit a decoder, evaluate another held-out bank, or change the earlier neuron-local precision optima.

## Main finding

At paired-response tolerance one, separately quantising the ten known centred output projections requires **77 bits per state in all three models**, or 154 bits for a pair. Its exact box certificate is below one in every output coordinate.

| Model seed | Certified optimum for the neuron-local precision bound | Output-projection report | Paired output-projection report | Largest certified paired error |
|---|---:|---:|---:|---:|
| 17 | 218 bits/state | **77 bits/state** | 154 bits | 0.988321129 |
| 29 | 230 bits/state | **77 bits/state** | 154 bits | 0.980876435 |
| 43 | 211 bits/state | **77 bits/state** | 154 bits | 0.961297677 |

The earlier optima remain valid: they concern separate quantisation of native hidden coordinates with the stated absolute-readout-gain bound. This control changes the permitted encoder. It demonstrates that a numerical response code can be much shorter when the encoder directly computes the target's output projections. It does not establish a more informative semantic mechanism.

## Encoder and exact range calculation

For normalised hidden activations `h_j∈[0,M_j/65536]`, let the known centred readout vectors be `c_{jk}=C_{jk}/(10·4096)`. Define the encoder's ten projections

\[
p_k(h)=c_k^\top h.
\]

The target is the centred-logit change `p_k(h′)−p_k(h)`. Any fixed output bias cancels. On the declared hidden box, each projection has the exact box range

\[
\ell_k=\frac{\sum_jM_j\min(C_{jk},0)}{D},\qquad
u_k=\frac{\sum_jM_j\max(C_{jk},0)}{D},
\qquad D=2684354560.
\]

Each extremum is attained on the mathematical box by choosing the appropriate endpoint of every hidden coordinate. Reachable network states may occupy a smaller subset, so the box range may remain conservative for the actual network. Its width is

\[
w_k=u_k-\ell_k=\frac{\sum_jM_j|C_{jk}|}{D}.
\]

Choose the minimum nonnegative integer `b_k` such that `w_k2^{−b_k}≤ε`. Equivalently,

\[
b_k=\max\{0,\lceil\log_2(w_k/\varepsilon)\rceil\}
\]

when `w_k>0`, with zero bits for zero width. The implementation uses exact comparisons of integers and fractions, not floating-point logarithms. It verifies both that the chosen bit depth satisfies the bound and that one fewer bit fails whenever `b_k>0`. These are exact componentwise optima for the ten independent projection-width constraints; no global claim about all possible response codes is made.

With `2^{b_k}` midpoint bins, let

\[
q_k(h)=\min\left(2^{b_k}-1,
\left\lfloor\frac{2^{b_k}(p_k(h)-\ell_k)}{w_k}\right\rfloor\right),
\]

with the zero-width case assigned code zero. The upper endpoint is clipped into the last bin. For every state in the box, midpoint reconstruction has error at most `w_k2^{−b_k}/2`. Therefore the simple paired decoder

\[
g_k(q(h),q(h'))=w_k2^{-b_k}\bigl(q_k(h')-q_k(h)\bigr)
\]

has error at most `w_k2^{−b_k}≤ε` by the triangle inequality. The interval offset and half-bin offset cancel in the difference. This proves the certificate on the whole stated box, without an empirical test.

The exact bit vectors are:

- Seed 17: `(7,8,8,8,8,8,7,8,7,8)`.
- Seed 29: `(7,8,8,8,8,8,7,7,8,8)`.
- Seed 43: `(8,8,8,8,8,7,7,7,8,8)`.

Their sum is 77 in each case. The ten centred coordinates sum to zero before quantisation; this deliberately simple control does not exploit that redundancy. Independently reconstructed coordinates need not sum to zero exactly, but they satisfy the declared coordinatewise error criterion. Requiring additional algebraic consistency would define another coding problem.

## Where the computation and coefficients move

Both constructions use the same known 32-by-10 readout matrix. They place it in different parts of the explanatory system.

| Component | Neuron-local precision report | Output-projection report |
|---|---|---|
| Mixed-coordinate encoder matrix | None | 320 centred-readout coefficient slots, all nonzero here |
| Scalar quantisers | 32 native activation quantisers | 10 output-projection quantisers |
| Quantiser ranges | 32 upper limits, with lower limits zero | 10 lower limits and 10 widths |
| Paired decoder matrix | 320 dense readout coefficient slots; reconstruction scales can be fused into them | 10 nonzero diagonal scale coefficients |
| Main matrix coefficients across encoder and decoder | 320 | 320, plus the 10 simple decoder scales |
| Additional constants | Native quantiser widths and integer bit depths | Projection offsets, widths, and integer bit depths |

The 10 projection decoder scales are derived from the stored widths and bit depths; the native decoder's reconstruction scales are similarly derived. Shared constants should be counted once in a full encoded-description ledger. The stage-wise counts above show the location of the work without pretending that moving the readout into the encoder eliminates its 320 coefficients. Reporting only the 10 decoder scales would conceal most of the computation.

This comparison also does not justify ranking models by report bits alone. An admissible report grammar might forbid target-weight-dependent directions, allow only a small number of native groups, constrain encoder operations, or charge semantic interpretation separately. Under those restrictions the projection encoder may be disallowed or expensive even though its communicated code is short.

## Interpretation: target access can make sufficiency cheap

Here the projections are the nominated outputs of the downstream linear head, apart from a bias that disappears in differences. The encoder therefore performs the computation whose explanation is being assessed and then quantises its result. A successful small response code is expected under that allowance. It is a useful control for detecting an under-specified explanatory budget, not evidence that the code exposes why the network computes those responses.

The computed quantity is the ten-dimensional **centred-logit change at the stated units and tolerance**. A different target, output scaling, permitted side information, or intervention family can change the required code. The result should not be converted into an unqualified claim about how many bits explain the network.

The appropriate methodological conclusion is specific: report-class restrictions and encoder costs matter as much as decoder restrictions. The neuron-local optimisation supplies a valid optimum within its declared class. The 77-bit control shows how a larger class with explicit target access changes the comparison and why a shared total cost ledger is necessary.

## Exact verification and retained evidence

`target_projection_control.py` generated `target_projection_control.json` from the three frozen `precision-gain-1.npz` files. It verifies all signed lower and upper range sums, their absolute-gain widths, every chosen integer bit depth and its one-bit-smaller failure, and the resulting rational paired bounds. Input and script hashes, all range numerators, all ten decoder scales, and coefficient counts are retained. All checks passed. No numerical optimisation is needed.

The script uses NumPy only to read the stored integer arrays; every certificate and bit-depth decision uses Python integers and exact fractions. The runtime command was:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/admin/.venvs/gaia-tess-microlensing/bin/python \
  work/theory/target_projection_control.py
```

The separately inspected nonlinear extension contract already states a first-hidden-layer cut, a nonlinear downstream ReLU block, a charged 698-parameter precision decoder, and its exploratory status after earlier test inspection. Those are appropriate scope declarations. This control did not inspect that extension's test results or verify its completed computations. Allowing a nonlinear encoder to run the whole nominated downstream function would recreate the same target-access issue, with that computation charged to the encoder.
