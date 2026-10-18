# Independent review of the three-run Goodfire exported-value audit

**Decision: the frozen audit passes independent arithmetic and local provenance checks within its stated scope.** No numerical correction to `findings.md` is needed. The audit correctly qualifies the source's control-range sentence by time and outcome. It supplies a reproducible reconstruction of published summaries; it does not supply native-checkpoint replication, a new controlled experiment, or a demonstrated generalist-to-specialist mechanism.

This review was performed by the empirical-review agent, separately from the agent that produced this Goodfire audit. I read the audit's scope, code, findings, saved checks, provenance and local source documents. I wrote and executed a separate standard-library `Fraction` checker without importing or executing the original acquisition, audit or verification programs. No network request, model execution, fit, bootstrap draw or raw-response AUROC calculation occurred. I did not change the reviewed audit. Prior work by this reviewer elsewhere in the research programme does not make this an independent research team or an independent experiment.

## Exact reviewed inputs and execution

The reviewed root `manifest.json` has SHA-256 `2bde68b01eff9c0b4b9eb799b82a59cad352d11a6ab9e414cabb05479f8a6cd4`. All 26 files listed there passed byte-size and SHA-256 checks. The source `inputs/manifest.json` is additionally bound through `input_freeze.json`, giving 27 distinct bound files. Its hash is `66000b8d960b8bfe6067359b7d3be985f45d74d5d347dd197dbdbf4ea61b7bf6`. Thus its omission from the root manifest's direct listing is not an unbound-input gap. Every locally read bound input was checked again after replay. The reviewer receipt records all exact input identities, including all nine measurement JSONs and all three CSVs.

Other principal identities:

| File | SHA-256 |
|---|---|
| Reviewed `input_freeze.json` | `46e304ddd240d67501a981218fa325936ef753b3b081e296acbde53f8a4f2e74` |
| Reviewed `audit.json` | `760dffcde994aa2d199acc4e55eaffc980b5d3d2ecbc7f4eea569619e132e124` |
| Reviewed `findings.md` | `fd057efebeb301d2eae14df03d7c2eb8dd96a266ebe240e35ac54f87b2ae8693` |
| Independent `recompute.py` | `d1c43647c814a627b4e97e77519825e037f18a947ccffad5ae9347d06f589fa8` |
| Full independent `receipt.json` | `b6c52b231a9c7bacbf3f41ac2621b8b684cca4137c1c1e7f87ea722dbacfd9d6` |
| Independent `recomputed.json` | `1b97792278bdd6f94c5c2169dd2f48c7664c1516287e95477cdcbb98ef6c70ad` |

The full run passed **29,543 counted assertions**, with no failures. The receipt gives the complete category counts. These are software assertions, not independent observations: they include 9,456 non-null leaf checks, 4,416 native-name/index checks, 2,601 AUROC/interval range checks, 2,805 ablation CSV numeric-cell comparisons, 1,920 individual control-value comparisons, 1,500 cohort-ID/label matches, and repeated source-to-summary/CSV consistency checks. The count must not be interpreted as statistical strength. The reviewed audit's original 206 checks and its same-agent 1,368-check rational replay are separate from this count.

Execution used Python 3.14.7 at `/opt/homebrew/opt/python@3.14/bin/python3.14`, with its standard library only. Reproduction commands and receipts are under `work/goodfire_training_independent_review/`.

## Reproduced results

All nine fixed-coordinate curves, containing 144 scores, agree with their source JSON and CSV. The checker follows the final discovery-selected coordinate backward; it never substitutes a checkpoint's separately selected winner. The first adjacent high pair gives the following nominal-epoch brackets. The exported exact steps remain in the replay output.

| Run | Broad: previous → first high; confirmation | Ambulance: previous → first high; confirmation | Cab adjacent pair ≥0.80 |
|---|---|---|---|
| Tiny seed 0 | 4 → 5; 7 | 5 → 7; 8 | Absent |
| Tiny seed 1 | 2 → 3; 4 | 4 → 5; 7 | Absent |
| Wider seed 0 | 1 → 2; 3 | 4 → 5; 7 | Absent |

The broad threshold is 0.85; the two subtype thresholds are 0.80. No score equals its threshold, so strict versus inclusive comparison changes none of these results. Two sampled scores above threshold do not establish continuous persistence, a confidence interval on emergence, or the first appearance of any broad/subtype information. Final-coordinate selection also makes these particular trajectories conditional on terminal discovery selection. Literal coordinate identity alone does not establish stable semantics through training.

The 4,210 unique evaluation IDs have the stated class counts: 250 car-family images, including 25 ambulances and 25 cabs, plus four images from each of 990 remaining classes. All three 500-image intervention cohorts are labelled subsets with 25 ambulances, 225 other cars and 250 non-cars. The hard-road evaluation has 44 negatives across eleven classes. Coordinate decoding and block order agree: the fixed final ambulance coordinate precedes the fixed final broad coordinate in every run, and the ablated candidate precedes the ambulance coordinate. These checks establish recorded identity/order, not causal ancestry.

All **255 ablation rows** were reconciled, preserving one candidate and sixteen distinct same-block controls at each of fifteen run/time combinations. All **120 signed comparisons** (eight fields × fifteen times) reproduce exactly, including source-order control values, signs, ties, rank intervals, bounds and denominators. Post-ablation AUROC and its decrease are algebraically related, and the comparisons reuse images, units and training runs; these are not 120 independent tests.

| Outcome | Candidate within the sixteen-control range |
|---|---:|
| Contribution scale | 14/15 |
| Oriented specialist-response change, ambulance images | 14/15 |
| Oriented specialist-response change, other-car images | 12/15 |
| Specialist AUROC after ablation | 14/15 |
| Ambulance margin drop, ambulance images | 11/15 |
| Ambulance margin drop, other-car images | 12/15 |
| Whole-logit RMS change | 14/15 |
| Specialist AUROC decrease | 14/15 |

The two central exceptions occur at step 5,000, approximately epoch 2:

- Wider seed 0's ambulance response decrease is **−0.0965025350 SD**, below the control minimum **−0.0775872692**. Under the declared orientation this is a response increase after ablation.
- Tiny seed 1's AUROC decrease is **+0.0017777778**, above the control maximum **+0.0014222222**. Its post-ablation AUROC is **0.5192888889**, below the control minimum **0.5196444444**. The AUROC-after and decrease entries describe the same exception in opposite orientations.

All three final ambulance-response changes and final AUROCs are within their respective control ranges. At the final checkpoint, the response changes are +0.0139866062, −0.0051702834 and +0.0496325232 SD. Other final outcomes do have exceptions: tiny seed 0's ambulance-image margin drop exceeds its control maximum, and wider seed 0's contribution scale exceeds every control scale. All fifteen out-of-range field/time entries, including the redundant AUROC pair, are retained with exact decimals in `recomputed.json`; none was removed from the replay.

Controls are not a fixed longitudinal set: their five sets contain 46, 47 and 49 distinct units respectively. Adjacent sets share only 1–8 units. At initialization, response and AUROC changes tie at zero. The written final tiny-seed-1 AUROC decrease is +10⁻¹⁶ despite equality at displayed precision. This arithmetic is reproducible, but the last decimal does not establish a material change or greater precision in the underlying experiment.

The source's initial readout range (0.855–0.874), final broad/hard-road AUROCs, candidate names, final response table and onset table all agree at stated reporting precision. The weak hard-road scores and strong random-initialization readout remain relevant constraints on interpreting a broad score as a learned car concept.

## Provenance and claim limits

The local receipt records six additional successful public JSON requests and six unchanged copies from prior retrieval. I checked the allowed URL set, recorded status/final URL, copied-file identities and local bytes; I did not independently make or authenticate those HTTP requests. The earlier seven-file corpus also matches its recorded local hashes. Nonidentity with those seven files establishes that the nine measurement files are additional to that corpus, not that they are newly published.

The [source protocol](https://already-clean.goodfire.com/assets/early-training/PROTOCOL.md), [source report](https://already-clean.goodfire.com/assets/early-training/RESULTS.md) and [training-map metadata](https://already-clean.goodfire.com/assets/training-map/manifest.json) were reviewed as saved local source material. They report discovery-only selection of coordinates, polarity/pooling, candidates and scale-matched controls, with separate evaluation. The supplied evaluation IDs and labels are internally consistent. The exports do **not** expose the discovery activation arrays, the entire scale-matching candidate pool, split-construction implementation, training examples or full model execution. Accordingly, the review cannot certify disjointness, verify that the sixteen controls really are the nearest discovery-scale matches, recompute source AUROCs/intervals, or reproduce the interventions. Source-reported parameter counts and training-data availability are likewise not independently checkpoint-verified. The protocol's claimed freeze date is not authenticated preregistration.

The source report itself discloses corrected image-and-label mixup in fresh runs after the historical recipe used label-only mixing. The fresh runs are therefore not exact replications of the historical trainer. This source qualification must survive integration. Two tiny seeds and one wider seed also do not establish a population distribution or isolate a capacity effect.

**No blocking defect within the audit's present wording.** Three boundaries would become major interpretive defects if removed in a manuscript: (i) promoting a control-range/rank comparison into a p-value, equivalence result or evidence of no causal effect; (ii) calling exported-summary agreement native or independent experimental replication; and (iii) claiming these retrospectively selected trajectories demonstrate generalist-to-specialist consolidation. Scale-matched controls are neither randomized assignments nor an exchangeable null sample. Their range is descriptive. The audit preserves these boundaries correctly.

The strongest counterargument to presenting this supplement as a substantive new mechanism result is that the original source already reports developmental selectivity without identifying a privileged precursor. This review improves inspectability and verifies the complete exception inventory; it does not turn that negative/narrow mechanistic conclusion into new evidence of reuse. At most, the source measurements describe selected-coordinate temporal ordering and this one-candidate ablation comparison under the reported experimental procedures.

## Portable replay

The same unchanged checker was tested on `work/new_supplements_packaging/goodfire_training_audit` in explicit `--numerical-only` mode. It passed **29,491 assertions** and produced a byte-identical `recomputed.json`. Its receipt hash is `bbf790f6497abe5a80d3d58ce361337c85d033713a32a5ae85ff706db78ff962`.

That layout omits `inputs/PROTOCOL.md` and `inputs/RESULTS.md`. Numerical mode checks retained inputs against the original bindings and the factual table transcription retained in `audit.json`; it does not replay source-prose transcription, external prior-copy/corpus checks or HTTP-provenance reconciliation. Those omissions are explicit. Merely recovering the two prose files would not recreate the full review environment because the default full mode also reads the original local acquisition paths and seven-file corpus.

This review assigns no broader generality and makes no claim that the audit closes the paper's broader significance gap.
