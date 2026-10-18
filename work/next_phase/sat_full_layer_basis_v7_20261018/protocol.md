# Full-layer native-axis and readout audit, version 7

## Question

This study reuses the exact 7,252-formula bank from the protected all-seventeen
label study and the unchanged Palumbo et al. epoch-900 checkpoint. It asks two
questions left open by the selected-unit analysis:

1. Are assignment-aligned directions more concentrated in the checkpoint's 512
   native final-MLP coordinates than in orthogonal rotations of the same complete
   post-ReLU state?
2. How much do the seventeen published singleton-assignment units contribute to
   task logits on threshold hits, misses and false positives across the full bank?

The study estimates alignment prevalence only within this checkpoint and this
engineered bank. It does not estimate prevalence across models, establish a
training mechanism, test recursive development, or create capacity pressure that
the model did not have.

Bank construction was response-blind and semantic: it used fixed formula sampling,
truth tables, historical token exclusion and hash order, with no checkpoint
activations or logits. The bank nevertheless oversamples the sixteen covered
assignments and is not a natural task distribution.

## Fixed inputs and native extraction

The input is `selected_bank.npz` with SHA-256
`38a8980436b67d7af8b897b813f68ca23f5d72b088fbfde7c6d64d4e4529d35e`.
No formula is added, removed or reweighted. One CPU float32 pass saves, at the
last token, all 512 final-block MLP preactivations and post-ReLU activations, the
128-dimensional residual stream immediately before and after that MLP, and all
16 logits. It also saves the exact 512-by-16 linear coefficient matrix from
post-ReLU units to logits. The extraction must reproduce the earlier seventeen
unit columns and logits under `numpy.allclose(atol=1e-6, rtol=1e-6)`, preserve every
source-threshold event and task argmax, reconstruct the saved post-MLP residual from the saved
components within `1e-4 + 1e-5 |x|`, and pass an eight-row fresh-load replay.
The replay evaluates each row inside the same 256-row batch boundaries as the
initial pass. The pinned custom runtime has two blocks, no final normalization,
`W_out` shape `(128,512)`, `W_U` shape `(128,16)`, and no unembedding bias; these
facts are asserted before any formula forward and documented in
`runtime_bindings.md`. Exclusive attempt and failure ledgers make every initiated
native or analysis run visible.

## Assignment-alignment analysis

For each of the 32 truth assignments, a formula is positive when that assignment
satisfies all ten clauses and negative otherwise. AUROC uses average ranks for
ties. The native preactivation analysis reports, for every one of 512 units, its
best positive-direction assignment AUROC; for every assignment it reports the
best unit; and for each published pair it reports the nominated unit's rank among
all 512 units. This is a within-bank prevalence description, not a population
estimate.

The basis comparison operates on the complete 512-dimensional post-ReLU vector.
An orthogonal rotation of that vector is invertible and changes only coordinates,
whereas rotating preactivations before ReLU would change the nonlinear model.
Because random coordinate signs are arbitrary, every native and rotated axis is
scored by `max(AUROC, 1-AUROC)`. The fixed master seed is 2026101811. Generate 63
independent 512-by-512 Haar bases by QR decomposition of standard-normal matrices,
with the sign of each Q column fixed by the corresponding diagonal entry of R.
Use NumPy 2.4.2 float64 with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` all fixed at one. Retain the
SHA-256 hash of each pre-QR Gaussian matrix and each resulting Q, but not the
matrices themselves. Every one-to-one optimum carries a checked Hungarian
primal-dual certificate.

For the native basis and each rotation report, both on the full bank and as a
secondary analysis restricted to satisfiable formulas:

- the maximum-weight one-to-one assignment-to-axis matching mean for all 32
  assignments;
- the same statistic separately for the sixteen assignments represented by the
  published singleton labels and the sixteen uncovered assignments;
- the mean and minimum, over assignments, of the best reusable-axis score;
- the number of axes whose best assignment score is at least 0.90, 0.95, 0.98
  and 0.99.

The two primary statistics are the uncovered-sixteen and all-32 one-to-one
matching means on the full bank. Apply Holm correction at familywise alpha 0.05:
the smaller Monte Carlo p-value must be at most 0.025, then the larger at most
0.05. With 63 rotations these correspond to zero and at most two reference bases
at least as large as native. All other p-values are descriptive. For each statistic, the directional Monte Carlo
comparison is `(1 + number of rotations at least native)/(1 + 63)`.

The uncovered labels can co-occur with covered labels. Therefore compute a
descriptive exclusive-positive secondary for each uncovered assignment. Restrict
the universe to satisfiable formulas for which no covered assignment is a
solution; within that universe, label rows by whether the uncovered assignment
satisfies the formula. Each uncovered assignment must have at least 20 positives
and at least 20 negatives or
the preflight fails. Apply the same one-to-one matching to native and rotated
bases. A statement about assignments outside the published selection set is
licensed only if both primaries reject after Holm correction and the native
exclusive-positive statistic exceeds all 63 references. Even then the statement
must remain conditional on this engineered bank and checkpoint.

The frozen conclusion map is: if both primaries and the exclusive condition pass,
report native-basis concentration with the preceding bank/checkpoint condition;
if both primaries pass but the exclusive condition does not, do not claim evidence
beyond the published selection set; if exactly one primary passes, report mixed
evidence and no general native-basis concentration; if neither passes, do not
claim native-basis concentration over the fixed references. The code-emitted
`primary_inference.licensed_conclusion` is the authoritative outcome text and
must be reported verbatim. Every branch must
state that this is a within-checkpoint, within-bank single-axis-separability
comparison against Haar references whose exchangeability null is generically
false for sparse nonnegative ReLU coordinates; it cannot separate assignment
specificity from generic coordinate privilege and does not establish causal use.

The Haar test asks whether native coordinates attain higher label-AUROC matching
than random orthogonal coordinates of the same raw post-ReLU state. Because ReLU
makes native coordinates nonnegative and sparse while rotated coordinates are
dense and signed, the exchangeability null is generically false. A native win
cannot by itself separate assignment-specific alignment from generic coordinate
privilege. Therefore the identical native-versus-Haar gap is reported
descriptively for 40 clause-presence labels and separately for SAT,
exactly-one-solution and multiple-solution labels. These are task-structured
control labels, not generic controls. The conclusion
must state this limit even if the assignment primaries reject.

This is a within-checkpoint axis-alignment comparison, not a classical randomised
experiment or evidence that any aligned coordinate is causally used.

## Final-readout analysis

For unit `j`, its exact additive 16-logit vector on input `i` is
`post[i,j] * (W_out[:,j]^T W_U)`. Report its SAT-minus-UNSAT contribution for
every published unit, split into true-positive, false-negative, false-positive
and true-negative threshold events at the source threshold 0.5. Exact zero
ablation subtracts that vector from the saved logits and reports all argmax
changes, correct-to-incorrect changes and incorrect-to-correct changes.

The resample control uses 64 deterministic replicates, master seed 2026101812,
with one NumPy `SeedSequence` child per published unit. For each unit and
replicate, independently replace its activation by a uniformly sampled
derangement within each of its five semantic strata: sole-positive,
multi-positive, UNSAT-negative, sole-negative and multi-negative. Rows are in
saved unique-ID order. This preserves the unit's empirical marginal distribution
inside each stratum while breaking formula-specific pairing. Report task accuracy,
argmax-change rates and correctness transitions overall and by threshold-event
category. Accuracy means argmax over all 16 logits equals token 13 for SAT or
token 14 for UNSAT. Also report exact zero ablation for all 512 units as a
reference distribution, joint zero ablation for units 55 and 416, and joint zero
ablation for all seventeen published units. Resampling one
last-layer unit has an exact linear downstream effect; it does not establish an
on-manifold or upstream intervention.

Compute and save `W_out.T @ W_U` in float64 before any forward, recompute it in
float64 before intervention statistics, require maximum disagreement at most
`1e-6`, and use the float64 matrix thereafter. Report the number of rows
whose top-two logit gap is below `1e-4` at baseline and after every zero ablation.
Before the first forward, independently verify the coefficient with `einsum` and
record its hash, all three weight hashes, the host name, processor and machine,
Torch CPU capability, and Torch build-configuration hash in the forward marker.
Run the native stages on the same physical host as the protected prior cache.
The frozen host and Torch-build fingerprint must match before any native attempt
ledger is created.

This resample preserves both the target-assignment class and the SAT/UNSAT class.
A small effect therefore concerns formula-specific variation within a semantic
stratum, not the importance of the assignment signal. TP/FN/FP/TN comparisons are
conditioned on the same baseline activation that is changed: their differences
are descriptive and must not be interpreted as causal effects of crossing 0.5.

## Dependence and stopping

The source threshold event is final-block post-ReLU activation `>=0.5`; expected
positive means the named assignment satisfies the formula, including multi-solution
formulas. Base-bank event counts must reproduce the prior protected analysis for
all seventeen units.

All formulas, assignments, units and rotated coordinates share one checkpoint.
The 63 rotations are reference bases, not independent models. The bank was
engineered around sixteen published assignments, so results for all 32 assignments
must be reported separately from the source-covered subset.

Before a new model is loaded, a frozen bank preflight verifies token-to-formula
decoding, assignment classes, the 43 task-structured control-label classes,
exclusive-positive counts,
bit order through every stored stratum, row IDs, resample feasibility and the
prior batch/thread environment. The live preflight must equal that frozen receipt.
Caught execution failures leave a failure ledger;
these local files are chronology evidence rather than an external append-only log.

Stop without relaxing the protocol on a source hash mismatch, environment drift,
non-finite value, missing class, failed reconstruction, mismatch with the earlier
seventeen-unit cache, replay mismatch, non-orthogonal rotation beyond `1e-10`, or
incomplete output. A defect found before any new model forward may be corrected
only in a new version with a new preflight, self-check, freeze and static review,
while preserving the rejected packet. Immediately before the first inference,
write an exclusive `forward_started_<stage>.json` marker. Once it exists, a
failure permanently stops that frozen packet; do not patch and resume it. An
attempt with neither receipt nor failure counts as post-forward if its marker
exists. Failures raised before a stage's attempt ledger is created—freeze
verification, environment and analysis thread checks, the replay initial-receipt
prerequisite, the native host-fingerprint check, and the analysis receipt
prerequisites—read no new-model output and may be retried in the same packet.
Every failure after attempt creation follows the marker rule; after any
`forward_started_*` marker exists, such failures are terminal. A native-stage
failure after its attempt but before its marker requires a new frozen version
because the exclusive attempt file prevents same-packet retry. Replay failures
after the initial marker and all failures after an analysis attempt are terminal.
Model-generated
static review is process evidence rather than human peer review.
