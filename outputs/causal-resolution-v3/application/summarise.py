#!/usr/bin/env python3
"""Report already-frozen experiment outcomes; no fitting or model selection."""
from pathlib import Path
from fractions import Fraction
import csv,json,hashlib,sys
import numpy as np
import experiment as ex

P=Path(__file__).resolve().parent
R=json.loads((P/'test_results.json').read_text())
PR=json.loads((P/'precision_test_results.json').read_text())
PD=json.loads((P/'precision_discovery.json').read_text())
D=json.loads((P/'discovery_summary.json').read_text())
CS=json.loads((P/'context_sensitivity.json').read_text()) if (P/'context_sensitivity.json').exists() else None
rows=[]
for r in R['results']:
    for bn,b in r['banks'].items():
        for method,v in b['results'].items():
            row={'seed':r['seed'],'bank':bn,'method':method,'n_rows':b['n_rows'],'mse_sum_outputs':v['prediction']['mse_sum_outputs'],'rmse_per_output':v['prediction']['rmse_per_output'],'max_absolute_error':v['prediction']['max_abs'],'fraction_within_epsilon1':v['prediction']['within_epsilon']['1.0'],'report_pair_oracle_empirical_mse':v['report_pair']['oracle_empirical_mse'],'report_pair_exact_minimax':v['report_pair']['exact_minimax_report_error'],'report_pair_codes':v['report_pair']['unique_codes'],'signed_difference_oracle_empirical_mse':v['signed_difference']['oracle_empirical_mse'],'signed_difference_exact_minimax':v['signed_difference']['exact_minimax_report_error']}
            row.update({f'fraction_within_epsilon{eps}':v['prediction']['within_epsilon'][str(float(eps))] for eps in [.25,.5,1.,2.,4.]})
            rows.append(row)
with (P/'all_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def mean_metric(bank,method,metric):
    return np.mean([r['banks'][bank]['results'][method]['prediction'][metric] for r in R['results']])

def fmt(x):return f'{x:.3f}'

lines=['# Trained-system application: finite reports, held-out intervention transfer, and precision repair','',
'This is a bounded executed application on three trained digit classifiers. It is not a replication of the Already-Clean experiments, an SAE comparison, a semantic-legibility assessment, or evidence about foundation-model prevalence. All originally planned methods were frozen before locked test evaluation. Two supplementary analyses—the affine-dual audit performed separately and the precision-certified native-state report—were designed after discovery results but also frozen before the test was opened.','',
'## Experimental contract and selection protection','',
'The preserved 1,797-image scikit-learn digits dataset was stratified once (seed 20261015) into 1,074 training, 238 discovery-fit, 121 discovery-validation and 364 locked test images. The three model initialisations were fixed as 17, 29 and 43. Each model has 64 inputs, 32 ReLU hidden units and ten output logits, and was trained for 200 fixed Adam epochs without early stopping. Training and fitting use NumPy only. Writer identities are absent: this random image split is not a writer-independent evaluation.','',
'Every weight and bias was exported as an integer divided by4096. Input pixels are integers in[0,16]. The hidden-state integer is h_int=max(X_int W1_int+16b1_int,0), representing h_int/65536. Responses are the actual ten centered-logit changes of this exported model. With quarter-strength interventions, every response is an exact integer divided by10737418240. This certifies the exported model, not a transcendental softmax probability. The unexported training network is retained to quantify export effects.','',
'The original contract and code were sealed in freeze.json; its SHA-256 is '+R['freeze_sha256']+'. Test evaluation began at '+R['test_started_utc']+'. The precision supplement was sealed separately in precision_freeze.json (SHA-256 '+PR['precision_freeze_sha256']+'). No locked outcome was used to choose reports, thresholds, ridge parameters, precision bits, or the four-predicate repair. These are local prospective freezes, not external preregistrations.','',
'Four fixed disjoint groups each contain eight hidden units. Discovery interventions attenuate one group by strengths1/2 and1. Locked tests include those same strengths on new images and the previously unused strengths1/4 and3/4. Each intervention bank contains364×4×2=2,912 rows per seed. A third, separate bank contains364 cyclic pairs of distinct natural held-out states and asks about replacing the entire state. Controlled attenuations need not correspond to natural inputs; natural-pair responses are a different transfer family.','',
'## Frozen finite reports and restricted decoder','',
'The reports are native-unit median predicates, greedy coactivation-paired sum predicates, and training-PCA direction median predicates. All medians and PCA/group constructions use training states. The paired-group implementation greedily chooses the largest remaining correlation and can end with a non-positive or degenerate pair; pre_test_notes.md records the correction to the original contract wording. PCA is a learned linear-direction baseline, not an SAE. No report is assigned a semantic name.','',
'Each family is evaluated at4,8,12 and16 bits per state. Candidate ordering is greedily selected by discovery-fit response mean-square error under ridge0.01. The fitted decoder is a single shared map B[T(h′)−T(h)] for all groups, strengths and natural pairs. It receives no input identity, digit label, intervention id or full hidden state. Its10B coefficients use binary64. Ridge is selected from a fixed grid on the separate discovery-validation set; the decoder is fit on discovery-fit only. Its additive signed-difference restriction is stronger than allowing an arbitrary function of the complete report pair.','',
'The four-bit repair starts from native8 and greedily adds native predicates from training quartiles, using discovery-fit MSE. The cost-matched comparison is native12, which simply continues the original median-predicate greedy ordering. Repair success was not assumed.','',
'| State report | Pair transmission | Nonzero encoder weights | Thresholds | Shared response coefficients |','|---|---:|---:|---:|---:|',
'| Native B predicates |2B bits|B|B|10B|',
'| Paired-group B predicates |2B bits|2B|B|10B|',
'| PCA B predicates |2B bits|at most32B|B|10B|',
'| Native quartile repair12 |24 bits|12|12|120|','',
'Dense projection coefficients are retained as integers with scale4096 and maximum absolute value at most4096. Native and paired-group coefficients are1. All thresholds are integer values of the training projection. This resource ledger shows why equal transmitted bit counts do not mean equal encoder cost.','',
'## Model and export checks','',
'| Seed | Test accuracy | Export/float class agreement | Largest centered-logit export error |','|---|---:|---:|---:|']
for r in R['results']:
    lines.append(f"|{r['seed']}|{100*r['export_accuracy']:.2f}%|{100*r['export_float_prediction_agreement']:.2f}%|{r['export_float_max_logit_error']:.6f}|")
lines += ['', 'All three models reached100% training accuracy. These test accuracies describe the classification task, not the adequacy of any finite report. There are only three model seeds, and intervention rows sharing an image are dependent. We report each seed and descriptive averages without treating rows or seeds as a large independent sample.','',
'## Main held-out intervention results','',
'MSE is the mean **sum over ten outputs** of squared centered-logit errors. The primary uniform tolerance is one centered-logit unit.','',
'| Seed | native8 MSE | native12 MSE | quartile repair12 MSE | native16 MSE | repair reduction vs native8 |','|---|---:|---:|---:|---:|---:|']
for r in R['results']:
    b=r['banks']['unseen_strength']['results'];v=[b[n]['prediction']['mse_sum_outputs'] for n in ['native-8','native-12','repair-12','native-16']]
    lines.append(f"|{r['seed']}|"+'|'.join(fmt(x) for x in v)+f"|{100*(1-v[2]/v[0]):.1f}%|")
lines += ['','The quartile repair improves MSE relative to native8 in all three models, but beats cost-matched native12 only in seed17. It is therefore a partial average-error improvement, not evidence for a generally superior repair rule. No low-bit report/decoder achieves the primary uniform tolerance.','',
'| Method | Seen-strength mean MSE | Unseen-strength mean MSE | Natural-pair mean MSE |','|---|---:|---:|---:|']
for n in ['native-4','native-8','native-12','repair-12','native-16','group-8','group-16','pca-8','pca-16']:
    lines.append('|'+n+'|'+'|'.join(fmt(mean_metric(b,n,'mse_sum_outputs')) for b in ['seen_strength','unseen_strength','natural_pairs'])+'|')
lines += ['','The small-strength transfer bank has smaller target effects than the seen-strength bank, so its lower raw MSE does not by itself imply stronger generalisation. Natural whole-state replacements produce much larger target changes and expose poor transfer of the discovery-fitted response decoder. None of the natural-pair predictions is within the primary tolerance in every output coordinate. These negative results are retained.','',
'The original intervention banks change the classifier decision on'+', '.join(f"{100*r['banks']['seen_strength']['class_flip_fraction']:.2f}%" for r in R['results'])+' of seen-strength rows, and'+', '.join(f"{100*r['banks']['unseen_strength']['class_flip_fraction']:.2f}%" for r in R['results'])+' of unseen-strength rows. The audit therefore includes interventions that affect model decisions, although a prediction change is not automatically harmful or semantically important.','',
'## Information loss versus decoder failure','',
'For each finite bank, group rows by the complete pair[T(h),T(h′)]. The exact smallest worst-case error of an **unrestricted bank-specific lookup decoder** is half the largest coordinate range inside any such fibre. The empirical squared-error floor is within-fibre conditional variance. Both quantities use test responses and are diagnostic oracles, not fitted or deployable explainers. A separate calculation groups signed differences alone; that coarser code diagnoses additional restrictions introduced by the chosen decoder input. Neither oracle is confused with the frozen B decoder.','',
'| Seed | native8 unseen actual MSE | Full report-pair empirical floor | Signed-difference empirical floor | Full report-pair exact minimax error |','|---|---:|---:|---:|---:|']
for r in R['results']:
    v=r['banks']['unseen_strength']['results']['native-8']
    lines.append(f"|{r['seed']}|{v['prediction']['mse_sum_outputs']:.3f}|{v['report_pair']['oracle_empirical_mse']:.3f}|{v['signed_difference']['oracle_empirical_mse']:.3f}|{v['report_pair']['exact_minimax_report_error']:.3f}|")
lines += ['','Some natural-pair reports have no repeated code at all: PCA8 has364 distinct report pairs in seeds29 and43, hence its finite-bank unrestricted lookup error is zero. Nevertheless, the actually fitted additive decoder has MSE826.269 and886.102 respectively. An injective finite code is therefore not an explanatory success under the declared computation restriction. The separate affine-dual audit certifies approximation obstruction without relying on decoder fitting; its results belong to that audit rather than being inferred from the gap here.','',
'At tolerance1, all three native8 unseen-strength banks contain exact report-pair collisions whose response separation exceeds2. This is a report-information failure regardless of fitting. The strongest exact counterexamples are stored in test_results.json together with row indices, code words, target coordinate and integer response numerators. Increasing a uniform response-error allowance from0 through1 weakens the definite-conflict tests at threshold2(1+δ) and expands possible conflicts at2(1−δ). These are certified outer brackets, not confidence intervals and not sharp identified sets. The true response allowance for the exported integer target is exactly zero.','',
'## A constructive precision repair with a whole-domain certificate','',
'A stronger numerical refinement quantises every hidden coordinate over a certified box. For all x_i∈[0,16], set M_j=max(0,16[Σ_i max(W1_int[i,j],0)+b1_int[j]]). Then0≤h_int,j≤M_j. Every coordinate attenuation by any strength in[0,1] remains in this box. Using2^{b_j} equal bins and midpoint reconstruction bounds each endpoint error by M_j/(2·65536·2^{b_j}). After applying the fixed centered linear head, the paired-response error in outputk is at most', '',
'Σ_j M_j |10W2_int[j,k]−Σ_lW2_int[j,l]| /(65536·10·4096·2^{b_j}).','',
'All displayed bounds are sums of exact rational numbers. Uniform precision takes the least common b satisfying every output bound. A deterministic gain-aware allocator starts at zero bits and adds the bit that most reduces the largest current bound, with exact tie rules. It uses only frozen model weights and the input domain, not discovery/test values. The greedy allocation is not claimed to minimise total bits.','',
'| Seed | Uniform bits/state | Gain-aware bits/state | Certified max error | Observed unseen max error | Lossless integer hidden box | Lossless quarter-state box |','|---|---:|---:|---:|---:|---:|---:|']
for pd,pr in zip(PD,PR['results']):
    a=pd['configs']['gain-1'];u=pd['configs']['uniform-1'];obs=pr['banks']['unseen_strength']['gain-1']['max_abs'];lossless=sum(int(m).bit_length() for m in pd['m_integer']);quarter=sum(int(4*m).bit_length() for m in pd['m_integer'])
    lines.append(f"|{pd['seed']}|{u['state_bits']}|{a['state_bits']}|{a['max_bound']:.6f}|{obs:.6f}|{lossless} bits|{quarter} bits|")
lines += ['','The lossless comparisons are coordinatewise fixed-length codes for the certified integer boxes, not lower bounds on an optimal joint code. The quarter-state box accounts for the fixed evaluated strengths; a continuous hidden domain has no finite exact lossless code. The certified schemes use a32-coordinate arithmetic encoder and320 fixed centered-readout coefficients. Pair transmission doubles the listed state cost. Their extra precision provides a real upper certificate, but it does not make them human-interpretable semantic accounts.','',
'The primary gain-aware scheme meets epsilon1 exactly on every evaluated response in all three banks and every seed. Its analytic guarantee is stronger: it covers the full declared hidden box and arbitrary coordinate attenuations. The code also evaluates epsilon0.25,0.5,2 and4 and preserves every bit allocation, rational bound and observed finite error. The largest observed error is far below the certified bound because the independent coordinate box is conservative.','',
'## Tolerance, budget and reporting limits','',
'Complete budget/tolerance results are in all_metrics.csv, precision_discovery.json and precision_test_results.json. Lower bit-budget MSE rankings do not settle explanatory economy once dense encoder cost is charged. Changing tolerance alters passing fractions but does not make the tested4–16-bit methods satisfy the primary uniform contract. The precision baseline gives a costed sufficient construction rather than extrapolating from an absence of observed collisions.','',
'This application demonstrates the value of separating report information, decoder capability, fitted performance and certified precision. It does not show that the modest native/group/PCA family is exhaustive, that PCA is inherently inferior, that a better nonlinear decoder could not transfer, or that coarse semantic descriptions fail in modern networks. The affine decoder, training objective, data size, random image split, particular intervention family and box bounds all limit the inference. Additional semantic-feature baselines, writer-held-out data, nonlinear downstream networks and larger pretrained systems remain untested.','',
'## Reproduction and retained files','',
'The NumPy-only experiment code is experiment.py. The original phases are prepare, fit and evaluate; each refuses to overwrite its stage outputs. To reproduce in another directory, copy the scripts and preserved digits.csv.gz and point preparation at the preserved input (the original source path is recorded). The saved model/report files are sufficient for exact evaluation without retraining. precision_repair.py provides supplementary freeze/evaluate phases. verify.py independently replays saved targets and predicates using Python integers. The frozen receipts prove content stability between stages within this run; they are not external timestamps or independent scientific replication.','',
'Dataset: Alpaydin,E. and Kaynak,C.(1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository, DOI https://doi.org/10.24432/C50P49, CC BY4.0. The retained1,797-image subset is the file distributed with scikit-learn1.9.0; see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html. The original UCI dataset is larger and includes source writer partitions that this bundled subset does not preserve as metadata.']
if CS:
    position=lines.index('## A constructive precision repair with a whole-domain certificate')
    unseen=[r for r in CS['records'] if r['bank']=='unseen_strength']
    lines[position:position]=['## Intervention-context sensitivity (post-test diagnostic)','',
        'A descriptive post-test exact oracle check supplies the intervention group and strength to the unrestricted report-pair decoder. This costs three extra context bits within either two-strength bank (four for the union of all four strengths). No fitted decoder is changed or selected. The resulting native8 minimum worst-case errors on unseen strengths remain '+', '.join(f"{r['exact_minimax_report_error']:.4f}" for r in unseen)+' across seeds17,29,43, all above the primary tolerance. Each retained witness compares two different images under the same group and strength. Thus the information-loss conclusion is not solely an artefact of withholding the action identity. This sensitivity was designed after test results and is reported as such.','']
repro_index=next(i for i,s in enumerate(lines) if s.startswith('The NumPy-only experiment code is'))
lines[repro_index]='The NumPy-only experiment code is experiment.py. Run `python3 reproduce.py /absolute/path/to/a/new-empty-directory` to regenerate models, report selection and both evaluations from the preserved exact split files, without an external data path, download, scipy or sklearn. Set BLAS thread counts to one as in the recorded run. The original phases are prepare, fit and evaluate; each refuses to overwrite its stage outputs. The original prepare phase records the local source path, while reproduce.py copies the preserved inputs. The saved model/report files also support exact evaluation without retraining. precision_repair.py provides supplementary freeze/evaluate phases. verify.py independently replays saved targets and predicates using Python integers. The frozen receipts prove content stability between stages within this run; they are not external timestamps or independent scientific replication.'
(P/'report.md').write_text('\n'.join(lines)+'\n')

def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    colors={'native':'#147d92','group':'#b36821','pca':'#7851a9'}
    for ax,bn,title in zip(axes,['seen_strength','unseen_strength','natural_pairs'],['Seen strengths, new images','Unseen strengths, new images','Natural whole-state replacement']):
        for fam,col in colors.items():
            vals=np.array([[r['banks'][bn]['results'][f'{fam}-{b}']['prediction']['mse_sum_outputs'] for b in [4,8,12,16]] for r in R['results']])
            ax.plot([4,8,12,16],vals.mean(axis=0),'o-',color=col,label=fam)
            ax.fill_between([4,8,12,16],vals.min(axis=0),vals.max(axis=0),color=col,alpha=.1)
        ax.set_title(title);ax.set_xlabel('Report bits per state');ax.set_xticks([4,8,12,16]);ax.set_ylabel('MSE, sum over ten logits')
    axes[0].legend(frameon=False);fig.savefig(P/'budget_transfer.png',dpi=180);fig.savefig(P/'budget_transfer.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for r,pr in zip(R['results'],PR['results']):
        sd=P/f"seed-{r['seed']}";pd=next(x for x in PD if x['seed']==r['seed'])
        axes[0].plot([.25,.5,1,2,4],[pd['configs'][f'gain-{e:g}']['state_bits'] for e in [.25,.5,1,2,4]],'o-',label=f"seed {r['seed']}")
        axes[1].plot([.25,.5,1,2,4],[pr['banks']['unseen_strength'][f'gain-{e:g}']['max_abs'] for e in [.25,.5,1,2,4]],'o-')
    axes[0].set_xscale('log',base=2);axes[0].set_xlabel('Certified error tolerance');axes[0].set_ylabel('Gain-aware bits per state');axes[0].legend(frameon=False)
    axes[1].plot([.25,.5,1,2,4],[.25,.5,1,2,4],'k--',label='tolerance');axes[1].set_xscale('log',base=2);axes[1].set_xlabel('Certified error tolerance');axes[1].set_ylabel('Observed unseen-strength maximum error');axes[1].legend(frameon=False)
    fig.savefig(P/'precision_cost.png',dpi=180);fig.savefig(P/'precision_cost.pdf');plt.close(fig)
    # Descriptive post-test witness; does not choose or change a report.
    r=R['results'][0];bn='unseen_strength';v=r['banks'][bn]['results']['native-8']['report_pair']['max_range_witness']
    if CS:v=next(r for r in CS['records'] if r['seed']==17 and r['bank']==bn)['max_range_witness']
    b=np.load(P/'seed-17'/f'test_bank_{bn}.npz');te=np.load(P/'test.npz');lookup={int(i):j for j,i in enumerate(te['ids'])}
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.8),layout='constrained')
    for ax,row in zip(axes,v['rows']):
        im,group,strength=b['meta'][row];j=lookup[int(im)];ax.imshow(te['x'][j].reshape(8,8),cmap='gray_r',vmin=0,vmax=16);ax.set_xticks([]);ax.set_yticks([])
        target=int(b['response_numerator'][row,v['coordinate']])/int(b['response_denominator'])
        ax.set_title(f'Image {im}, digit {te["y"][j]}\ngroup {group}, attenuation {strength}/4\noutput {v["coordinate"]} change: {target:.3f}')
    fig.suptitle('Same intervention and frozen report pair, incompatible responses\nSeed 17; post-test descriptive context-controlled collision',fontsize=11)
    fig.savefig(P/'collision_witness.png',dpi=180);fig.savefig(P/'collision_witness.pdf');plt.close(fig)
plots()
print(json.dumps({'report':str(P/'report.md'),'csv_rows':len(rows),'figures':['budget_transfer','precision_cost','collision_witness']}))
