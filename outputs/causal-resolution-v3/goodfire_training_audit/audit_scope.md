# Retrospective exported-value audit: scope fixed before additional retrieval

This bounded source audit covers all three fresh runs named in the retained
training-map manifest, in that order: `fresh-tiny-seed0`, `fresh-tiny-seed1`, and
`fresh-small-seed0`. It is retrospective. The report, protocol, manifest and the
already retrieved tiny-seed-0 exports have been read; this is not a preregistered
or blinded scientific test. The remaining six JSON files have not yet been
retrieved by this audit. Existing tiny-seed-0 bytes will be copied unchanged.

## Inputs and allowed retrieval

Retain exact bytes, source URLs, sizes and SHA-256 for the report, protocol,
manifest and each run's `metrics.json`, `lineage.json`, `causal.json`. Resolve only
the report's published `fresh-*/{metrics,lineage,causal}.json` artifact template
using the manifest's literal run IDs and public `assets/early-training/` base.
The six allowed additional URLs will be recorded before retrieval. Do not fetch
timeline binaries, images, weights, full probes, guessed paths or the already
403 `fresh-figures/summary.json`; do not try an access workaround. Retain any
retrieval failure and mark dependent checks unavailable. Freeze all acquired
inputs before the complete arithmetic audit.

## Fields and deterministic checks

1. **Provenance and denominators.** Record run IDs, parameters, units, block/name
   conventions, 16 declared update steps, evaluation IDs/labels, uniqueness and
   per-class counts, and each source selection/matching statement. Check the
   supplied 500-image intervention cohort against the manifest's 4,210-image
   evaluation cohort. Distinguish independently counted IDs from source-reported
   training size, split procedure and causal-selection implementation.
2. **Fixed-coordinate onsets.** Use `lineage.targets` and `lineage.rows`, never
   replace them with each checkpoint's newly selected `metrics` winner. For broad
   car use threshold 0.85; for ambulance and cab use 0.80. The onset is the first
   member of the first two consecutive exported rows at or above threshold. Keep
   the previous sampled epoch/step, first-high epoch/step and confirming sample;
   no interpolation or claim of a true emergence time. Missing/nonfinite scores
   cannot form a pair. Preserve threshold equality counts and also state whether
   the source prose's strict “above” wording changes any result. Record the full
   nine fixed-coordinate curves and all absent/undefined values.
3. **Coordinate identity/order.** Check mode/index decoding against the manifest,
   final lineage/metrics agreement, update/epoch alignment, the specialist's
   position relative to the final broad coordinate, and the nominated precursor's
   upstream restriction. This validates names/order, not biological ancestry or
   stable meaning of a coordinate through training.
4. **All intervention comparisons.** For each of three runs at each of five
   exported times retain the candidate and every control in source order. Expected
   denominator: one candidate plus sixteen controls, hence 255 exported ablation
   rows total. Retain unit identity, candidate flag, contribution scale, both
   oriented specialist-response changes, specialist AUROC after ablation, both
   ambulance-margin drops and whole-logit RMS change. Also compute signed AUROC
   decrease = exported pre-ablation AUROC minus exported post-ablation AUROC.
   For each numeric field report candidate value, all sixteen control values,
   valid/absent/null/nonfinite counts, minimum/maximum, sign counts, exact decimal
   ties, and numbers of controls lower/equal/higher than the candidate. Report both
   ascending and descending rank intervals (including ties), solely descriptive.
   Do not take absolute values, choose a favorable time, invent missing cells,
   pool ranks into a p-value, or call matched controls randomized replicates.
   Preserve whether control identities change across times.
5. **Report reconciliation.** Transcribe the two report tables with their stated
   precision and compare all entries against the exported fixed-coordinate and
   intervention numbers. Separately record initial readout AUROCs, final hard-road
   scores, sustained-cab absence, and the report's final control-range claim. The
   exact sampled epochs differ slightly from their round nominal labels; record
   both. Rounding checks use half a unit in the report's last decimal place.

Decimal comparisons operate on the written JSON numbers; equality is equality
at exported precision, not equality of unavailable raw responses. Missing fields,
nulls, nonfinite values, unexpected row counts and schema changes are explicit
findings, not zeroes. Arithmetic assertions verify internal consistency only.

## Boundaries and outputs

No training, neural inference, image retrieval, statistical fitting, bootstrap,
new confidence interval, or recomputation of AUROC from per-image responses.
Published AUROCs and intervals remain source-reported measurements. Source claims
about disjoint selection, scale matching and model execution cannot be reproduced
without their underlying arrays and implementation. The cited protocol's freeze
date is a source statement, not independently verified preregistration.

Use Python's standard library and one CPU thread. Save the allowlist/provenance,
input freeze, audit code, complete row-level comparison tables, check receipt and
a concise findings memo with primary links. No original output/archive or other
agent artifact is modified. The memo may establish internal agreement or expose
disagreement; it cannot establish causal inheritance, randomization significance,
population prevalence, or independent experimental replication.
