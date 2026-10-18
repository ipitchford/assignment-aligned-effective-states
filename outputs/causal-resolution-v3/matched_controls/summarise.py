from pathlib import Path
from fractions import Fraction as Q
import csv,json
P=Path(__file__).resolve().parent
d=json.loads((P/'discovery_results.json').read_text());t=json.loads((P/'test_results.json').read_text())
names={'native8_affine_minimax':'Native 8','native12_affine_minimax':'Native 12','contrast12_affine_minimax':'Contrast 12','greedy_minimax12':'Greedy minimax 12'}
methods=list(names)
disc={(r['seed'],r['method']):r for r in d['results']}
test={(r['seed'],r['bank'],r['method']):r for r in t['results']}
lines=['# Post-test exploratory matched minimax controls','',
'**Conclusion:** Contrast-guided selection does not consistently outperform matched native12 or direct greedy minimax selection. Its value here remains diagnostic certification rather than an empirically established superior repair rule. Every evaluated model still fails the primary worst-case tolerance of one.','',
'This analysis was commissioned after the original test outcomes and independent empirical review were known. Selection in this implementation uses discovery responses only, and its final decoders were frozen before this implementation evaluated the existing test banks. Those procedural steps do not restore pristine validation: these are exploratory comparisons on previously opened data. Original contracts, freezes, model files and review manuscript were unchanged.','',
'## Matched methods','',
'All four methods use an intercept, ten independent affine minimax output fits, the original uncompressed LP implementation for their final fit, and nearest-integer coefficient export over 2^32. Each 12-predicate method transmits 12 bits per state/24 per pair, uses 12 native threshold predicates and 130 dyadic decoder coefficients including the intercept. Native12 retains the originally selected median predicates. Contrast12 retains the original contrast-selected quartile predicates. Direct greedy minimax starts from native8 and, for each of four additions, evaluates **every remaining quartile candidate on all ten outputs**, selecting the lowest maximum objective with the declared numeric tie rule. No shortlist, candidate pruning or test-driven choice was used.','',
'For candidate search only, identical feature rows are collapsed to per-output minimum/maximum responses. The inequalities are exactly equivalent: a shared prediction must lie within t of both extrema and therefore every response in the group. Full-row and compressed objectives were checked at the initial design for each seed. The final selected sets were fitted and certified with the original full-row implementation. Greedy choices are numerical, not exact proofs of stepwise or global feature optimality.','',
'## Discovery worst-case error','',
'The table displays the exact primal upper bound rounded to four decimals. Every output has a retained rational lower certificate and dyadic primal upper certificate.','',
'| Seed | Native 8 | Native 12 | Contrast 12 | Greedy minimax 12 |','|---:|---:|---:|---:|---:|']
for seed in [17,29,43]:lines.append('| '+str(seed)+' | '+' | '.join(f'{float(Q(disc[seed,m]["joint_upper"])):.4f}' for m in methods)+' |')
lines += ['', 'Direct greedy minimax has lower discovery maximum than contrast selection for seeds 17 and 43 and ties seed 29. This does not imply that its four-step result is globally optimal over all 12-predicate sets.','',
'## Existing held-out banks: maximum error','',
'| Bank | Seed | Native 8 | Native 12 | Contrast 12 | Greedy minimax 12 |','|:--|---:|---:|---:|---:|---:|']
for bank in ['seen_strength','unseen_strength','natural_pairs']:
    for seed in [17,29,43]:lines.append('| '+bank+' | '+str(seed)+' | '+' | '.join(f'{test[seed,bank,m]["maximum_absolute_error"]:.4f}' for m in methods)+' |')
lines += ['', 'On unseen strengths, contrast selection beats native12 on maximum error in one seed and loses in two. Against direct greedy minimax it wins once, loses once and ties once. It does not dominate on either same-strength or natural-pair transfer. All methods remain far above tolerance one.','',
'## Existing held-out banks: mean squared error','',
'Values are average squared errors summed across all ten outputs; these are descriptive finite-bank quantities, not independent-replication estimates.','',
'| Bank | Seed | Native 8 | Native 12 | Contrast 12 | Greedy minimax 12 |','|:--|---:|---:|---:|---:|---:|']
for bank in ['seen_strength','unseen_strength','natural_pairs']:
    for seed in [17,29,43]:lines.append('| '+bank+' | '+str(seed)+' | '+' | '.join(f'{test[seed,bank,m]["mse_sum_outputs"]:.3f}' for m in methods)+' |')
lines += ['', 'Contrast selection has lower unseen-strength MSE than direct greedy minimax in all three seeds, despite its mixed maximum-error comparison; maximum and average objectives differ. Compared with native12, contrast MSE is lower only in seed 43 on unseen strengths. In seed 29, every minimax method has unseen-strength MSE exceeding a zero-response predictor, illustrating that an optimum for discovery maximum error need not optimise held-out average error. The complete CSV includes the normalised MSE.','',
'## Execution, exact certificates and limits','',
'| Seed | Candidate coordinate LPs | Four-step selection seconds | Search, final fits and certificates seconds |','|---:|---:|---:|---:|']
for r in d['timing']:lines.append(f'| {r["seed"]} | {r["candidate_lp_count"]} | {r["selection_seconds"]:.3f} | {r["total_seed_seconds"]:.3f} |')
ident=[disc[s,'contrast12_affine_minimax']['original_frozen_coefficient_numerators_identical'] for s in [17,29,43]]
lines += ['',f'The complete run used {sum(r["candidate_lp_count"] for r in d["timing"]):,} candidate coordinate LPs and {d["total_seconds"]:.3f} seconds for search, final fits and certificates, in one process with single-thread BLAS/HiGHS configuration. This is a local observed runtime, not a general complexity claim. '+('All three newly refitted contrast12 coefficient arrays exactly match their original frozen dyadic arrays, so its displayed predictions reproduce the original contrast results.' if all(ident) else 'Some contrast12 coefficient arrays differ from the original LP solutions; their matches and maximum numerator discrepancies are recorded in the discovery results.'),'',
'`verify.py` checks input/freeze integrity, exhaustive candidate identities, the declared selection tie rule, 120 exact discovery output brackets and 36 exact test maxima without invoking an optimiser. The precise brackets, selected predicates, all candidate objectives, coefficients, exact responses and runtime are retained in `discovery_results.json`, `seed-*`, and `greedy-*.json`. `verification.json` records the replay outcome.','',
'These controls close the missing matched-comparator gap descriptively. They do not turn the same small linear-readout study into evidence about semantic explanation quality, foundation models, prevalence or independent generalisation. The exact sparse witnesses still serve their rejection purpose. The present measurements do not justify marketing witness-contrast selection as the best four-predicate repair.','',
'Reproduce in a new empty directory with `run.py run`, followed by `run.py evaluate`, keeping its relative `../review-round1` input location or adjusting that constant. These stages refuse to overwrite their contract/freeze/results. Run `verify.py` for solver-free replay and `summarise.py` to regenerate this report and the CSV. The interpreter/environment and input hashes are retained in the exploratory contract.']
(P/'report.md').write_text('\n'.join(lines)+'\n')
with (P/'results.csv').open('w',newline='') as f:
    fields=['seed','bank','method','n','mse_sum_outputs','maximum_absolute_error','exact_max_error','fraction_all_outputs_within_1','mse_relative_zero_response']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(t['results'])
print('Wrote report.md and results.csv')
