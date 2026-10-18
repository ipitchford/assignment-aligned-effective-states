# Three fresh Goodfire runs: retrospective exported-value audit

**Finding.** The published onset brackets, final AUROCs and final response-change
table agree with the exported measurements for all three runs. The broader
statement that effects lie inside matched-control ranges needs a time/outcome
qualification: it holds for the final specialist-response and AUROC comparisons,
but not for every exported time or outcome. This is an arithmetic and provenance
audit, not independent experimental replication or evidence of causal inheritance.
[Source report](https://already-clean.goodfire.com/assets/early-training/RESULTS.md).

## Inputs and what was independently checked

The [manifest](https://already-clean.goodfire.com/assets/training-map/manifest.json)
names two 91,828-parameter runs (`fresh-tiny-seed0/1`) and one 345,552-parameter run
(`fresh-small-seed0`, labelled “Wider”). These are fresh ten-epoch runs, distinct
from the epoch-90 medium/large microscopes. Six additional public JSON requests
succeeded: metrics, lineage and causal exports for tiny seed 1 and wider seed 0.
The tiny-seed-0 files, report, protocol and manifest were copied unchanged from
the prior retrieval. No binaries, images, checkpoints or unavailable summary URL
were fetched. None of the nine measurement-file hashes matches the original
seven-file source corpus; this means new to this audit, not newly published.

The audit counted 4,210 unique evaluation IDs: 25 per car class (250 total), plus
four from each of 990 other classes. Each intervention file contains 500 unique
IDs with matching manifest labels: 25 ambulances, 225 other cars and 250 non-cars.
The eleven hard-road classes supply 44 negatives. All three exports contain 16
trajectory samples and five intervention times, with one nominated candidate and
16 distinct same-block control units at each time. That is **144 fixed-coordinate
score rows and 255 ablation rows**, not 255 independent experiments. All 120
candidate/control field comparisons retain exact written signs, ties, control
identities and denominators. No missing, null or nonfinite cells were found in
the audited source JSON; full per-image responses are not supplied by these files.

The source says that final coordinates, their polarity/pooling, the early
upstream candidate and scale-matched controls were selected on discovery images.
The audit preserves those statements and checks coordinate decoding and cohort
membership. It cannot verify the selection implementation, train/evaluation split
construction, scale computation, or that these were the sixteen nearest controls
without the omitted underlying arrays. The protocol's 20 September freeze date
is source-reported, not independently authenticated preregistration.
[Protocol](https://already-clean.goodfire.com/assets/early-training/PROTOCOL.md).

## Fixed-coordinate onset reconstruction

The first-high sample must be followed by another sampled score at or above 0.85
(broad) or 0.80 (ambulance/cab). These are the final discovery-selected coordinates
followed backward, not each snapshot's newly selected winner. “Confirm” below is
the second high sample. Epoch labels in the table are nominal rounded labels;
exact steps/epochs and all scores are retained in `fixed_coordinate_curves.csv`.

| Run | Broad: previous → first high; confirm | Ambulance: previous → first high; confirm | Cab sustained 0.80 |
|---|---|---|---|
| [Tiny seed 0 lineage](https://already-clean.goodfire.com/assets/early-training/fresh-tiny-seed0/lineage.json) | 4 → 5; 7 | 5 → 7; 8 | Not reached |
| [Tiny seed 1 lineage](https://already-clean.goodfire.com/assets/early-training/fresh-tiny-seed1/lineage.json) | 2 → 3; 4 | 4 → 5; 7 | Not reached |
| [Wider seed 0 lineage](https://already-clean.goodfire.com/assets/early-training/fresh-small-seed0/lineage.json) | 1 → 2; 3 | 4 → 5; 7 | Not reached |

No exported score equals a threshold, so replacing “at or above” with the
protocol prose's strict “above” changes none of these results. The brackets locate
the first member of an adjacent high pair in a sampled curve. They are not
confidence intervals, a reconstruction between snapshots or a population onset.
The fixed final ambulance coordinate is earlier in network block order than the
fixed final broad coordinate in all three runs. The separate ablated candidate
is strictly upstream of its specialist; coordinate ordering is not an ancestry
proof.

Initial multi-coordinate readout AUROCs are source-reported as 0.873673, 0.857327
and 0.855230, reproducing the report's 0.855–0.874 range. Final broad/hard-road
AUROCs are 0.861171/0.506636, 0.926448/0.536636 and 0.970159/0.614864.
These values were extracted and reconciled, not recomputed from raw responses.
[Tiny 0 metrics](https://already-clean.goodfire.com/assets/early-training/fresh-tiny-seed0/metrics.json),
[tiny 1 metrics](https://already-clean.goodfire.com/assets/early-training/fresh-tiny-seed1/metrics.json),
[wider metrics](https://already-clean.goodfire.com/assets/early-training/fresh-small-seed0/metrics.json).

## Candidate versus every matched control

Positive response decrease follows the source's orientation convention. Negative
values mean the oriented specialist signal increased after the ablation; they
must not be converted to unsigned damage.

| Run at epoch 10 | Candidate response decrease, SD | Sixteen-control response range, SD | Specialist AUROC before → after |
|---|---:|---:|---:|
| [Tiny seed 0 causal](https://already-clean.goodfire.com/assets/early-training/fresh-tiny-seed0/causal.json) | +0.01398661 | −0.01353981 … +0.02177744 | 0.86364444 → 0.86435556 |
| [Tiny seed 1 causal](https://already-clean.goodfire.com/assets/early-training/fresh-tiny-seed1/causal.json) | −0.00517028 | −0.01875624 … +0.00945821 | 0.84924444 → 0.84924444 |
| [Wider seed 0 causal](https://already-clean.goodfire.com/assets/early-training/fresh-small-seed0/causal.json) | +0.04963252 | −0.06430776 … +0.07568929 | 0.93422222 → 0.93031111 |

All three final response decreases and final post-ablation AUROCs are within
their respective control ranges. Across all fifteen run/time comparisons, the
candidate lies in the response-change range in fourteen and the AUROC range in
fourteen. The two exceptions concern different runs:

- At approximately epoch 2, wider seed 0 has response decrease **−0.09650254 SD**,
  below the control minimum −0.07758727. This is an increase in the oriented
  response, not evidence for precursor-supported specialist activation.
- At approximately epoch 2, tiny seed 1 has AUROC decrease **+0.00177778**,
  above the control maximum +0.00142222. Its post-ablation AUROC is 0.51928889,
  versus a control minimum 0.51964444. These are descriptive exported differences.

Other outcomes also have exceptions; all are retained in the full comparison
table. Even at epoch 10, tiny seed 0's ambulance logit-margin drop is 0.00679465,
slightly above the control maximum 0.00654696. Its epoch-2 whole-logit RMS change
also exceeds all controls. Wider seed 0's final contribution scale is 0.04119045,
above the largest exported control scale, 0.02996745: “matched” must not be read
as equal scale. There is no post hoc normalization or significance test here.

Control identities change with time in every run: 46, 47 and 49 distinct control
units appear across the respective five sets. They are not a single longitudinal
cohort of sixteen. Epoch-0 specialist-response changes and AUROC changes tie at
zero across each candidate and all sixteen controls. Later exact exported AUROC
ties are preserved. The tiny-seed-1 final AUROC subtraction is +1×10⁻¹⁶ in the
written decimals, despite equal displayed values; it is retained as export-level
roundoff, not interpreted as a material decrease. No rank is a randomization
p-value, and no multiplicity-adjusted or causal claim follows from range extremes.

## Reproducibility and boundary

`audit_exports.py` performs 206 internal/report-reconciliation checks;
`audit_replay_receipt.json` confirms exact saved-result reproduction. A separate
`Fraction` implementation reproduces nine onset sequences and all 120 comparisons
with 1,368 mechanical checks. This is separate code by the same agent, not an
independent investigator or additional scientific observations. Both programs use
only Python's standard library on one thread. No models, training, new fits,
bootstrap or confidence-interval estimation were used.

Run `python3 -B audit_exports.py --verify` and `python3 -B verify_arithmetic.py`
from this directory. `audit.json` retains source statements, all signed comparisons
and checks; the three CSVs expose every audited row. `input_provenance.json` gives
all primary URLs and byte identities. `manifest.json` binds the completed audit.

Key SHA-256 identities: input freeze
`46e304ddd240d67501a981218fa325936ef753b3b081e296acbde53f8a4f2e74`;
audit code `8c676f2e64b28b1416bd7e5d0d53fe1820a75c5abb242463dd384c55408f52db`;
audit result `760dffcde994aa2d199acc4e55eaffc980b5d3d2ecbc7f4eea569619e132e124`.

The report explicitly places training checkpoints and complete probe arrays in
its devbox experiment directory. This audit does not establish whether a separate
public release exists. With the inspected exports, one can check ordering,
reported summaries and candidate/control contrasts. One cannot independently
recompute AUROC/intervals, reproduce native interventions, establish semantic
continuity through training, or infer a general causal consolidation mechanism.
