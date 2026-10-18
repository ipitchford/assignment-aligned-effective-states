#!/usr/bin/env python3
"""Reanalyse the frozen, author-reported Already Clean aggregates.
No model execution or per-image replication is claimed. All paths are project-relative.
"""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
from statistics import median
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/source'; OUT=ROOT/'data/derived'; RES=ROOT/'results'; FIG=ROOT/'paper/figures'
for p in (OUT,RES,FIG): p.mkdir(parents=True,exist_ok=True)
COL={'damage':'#A45162','repair':'#5D8B83','other':'#AAA185','net':'#493D67', 'grey':'#70736A'}

def csvwrite(path: Path, rows: list[dict]):
    if not rows: raise ValueError('No rows to write')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def wilson(k: int,n: int,z: float=1.959963984540054):
    p=k/n; d=1+z*z/n; mid=(p+z*z/(2*n))/d
    rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(mid-rad),100*(mid+rad)

def main():
    data=json.loads((SRC/'weak-ablation.json').read_text())
    rows=[]; selected=[]; failures=[]; interval_diffs=[]
    for model in data['models']:
        seen=set()
        for cell in model['cells']:
            n=cell['n'];b=cell['baseline_accuracy'];d=cell['damage'];r=cell['repair'];f=cell['flips']
            key=(cell['layer'],cell['percentile'],cell['arm'])
            checks={
                'unique_cell': key not in seen,
                'eval_n':n==data['evaluation_n'],
                'integer_counts':all(isinstance(v,int) for v in (d,r,f,n)),
                'baseline_count_integer':abs(b*n-round(b*n))<1e-7,
                'accuracy_identity':abs(cell['accuracy']-b-(r-d)/n)<1e-12,
                'delta_identity':abs(cell['delta_pp']-100*(r-d)/n)<1e-10,
                'flips_cover_transitions':0<=d+r<=f<=n,
                'damage_feasible':d<=round(b*n),
                'repair_feasible':r<=n-round(b*n),
                'wrong_to_wrong_feasible':f-d-r<=n-round(b*n)-r,
                'baseline_consistency':abs(b-model['baseline_accuracy'])<1e-12,
            }
            seen.add(key)
            failures += [{'model':model['key'],'arm':cell['arm'],'check':k} for k,v in checks.items() if not v]
            delta=(r-d)/n
            # Independent paired-normal comparator; the source CI method is not supplied.
            half=1.96*100*math.sqrt(((d+r)/n-delta*delta)/n)
            diff=max(abs(cell['delta_pp_95ci'][0]-(100*delta-half)),abs(cell['delta_pp_95ci'][1]-(100*delta+half)))
            interval_diffs.append(diff)
            lo,hi=wilson(f,n)
            acc=cell.get('accounting')
            energy=[a['squared_mass_selected_fraction'] for a in acc.values()] if acc else [cell.get('removed_l2_fraction',float('nan'))]
            row={'model':model['key'],'title':model['title'],'layer':cell['layer'],'percentile':cell['percentile'],'arm':cell['arm'],
                 'n':n,'baseline_accuracy_pct':100*b,'accuracy_pct':100*cell['accuracy'],'delta_pp':100*delta,
                 'damage':d,'repair':r,'flips':f,'wrong_to_wrong':f-d-r,
                 'damage_pct':100*d/n,'repair_pct':100*r/n,'disagreement_pct':100*f/n,'wrong_to_wrong_pct':100*(f-d-r)/n,
                 'gross_correctness_transition_pct':100*(d+r)/n,
                 'conditional_damage_pct':100*d/round(b*n),
                 'cancellation_fraction':1-abs(r-d)/(r+d) if r+d else 0.0,
                 'disagreement_wilson_lo_pct':lo,'disagreement_wilson_hi_pct':hi,
                 'source_delta_ci_lo_pp':cell['delta_pp_95ci'][0],'source_delta_ci_hi_pp':cell['delta_pp_95ci'][1],
                 'source_kl_definition_unverified':cell['kl'],
                 'removed_l2_median_pct':100*median(energy),'removed_l2_min_pct':100*min(energy),'removed_l2_max_pct':100*max(energy),
                 'intervention_scope':'final layer only' if model['key']=='inception' else 'all 18 blocks' if cell['layer']=='all' else 'single block',
                 'paired_interval_reconstruction_max_abs_pp':diff}
            rows.append(row)
            if cell['layer']==('mixed5b' if model['key']=='inception' else 'all'):selected.append(row)
    csvwrite(OUT/'ablation_all_345.csv',rows)
    csvwrite(OUT/'ablation_selected_20.csv',selected)
    csvwrite(OUT/'ablation_p75.csv',[r for r in selected if r['percentile']==75])
    assert len(rows)==345 and len(selected)==20
    assert not failures, failures
    # Illustrative analyst-chosen frontier: absolute net accuracy loss at most 1pp;
    # prediction disagreement at most 5%. This is illustration, not a normative threshold.
    frontier=[]
    for m in data['models']:
        mr=[r for r in selected if r['model']==m['key']]
        passes_acc=[r['percentile'] for r in mr if abs(r['delta_pp'])<=1]
        passes_both=[r['percentile'] for r in mr if abs(r['delta_pp'])<=1 and r['disagreement_pct']<=5]
        frontier.append({'model':m['key'],'max_tested_percentile_abs_delta_le_1pp':max(passes_acc,default=0),
                         'max_tested_percentile_also_disagreement_le_5pct':max(passes_both,default=0)})
    csvwrite(OUT/'illustrative_tolerance_frontier.csv',frontier)
    toy=json.loads((SRC/'toy-pca-density.json').read_text()); trows=[]
    for key,value in toy.items():
        c,v,h,p=key.split('|')
        trows.append(dict(correlated=int(c),variant=v,hidden=int(h),density=float(p),source_group_share=value['s'],source_sd=value['sd']))
    trows.sort(key=lambda r:(r['correlated'],r['variant'],r['hidden'],r['density']))
    csvwrite(OUT/'source_toy_96.csv',trows)
    assert len(trows)==96 and all(0<=r['source_group_share']<=1 and r['source_sd']>=0 for r in trows)
    stats=json.loads((SRC/'figure-data.json').read_text())['stats']
    assert stats['a']+stats['b']+stats['c']==stats['judged']
    report={'status':'PASS: aggregate arithmetic only','source_date':data['source_date'],'source_verification_date':data['verification_date'],
            'evaluation_n':data['evaluation_n'],'calibration_n':data['calibration_n'],'intervention_cells':len(rows),
            'checks_per_cell':11,'arithmetic_checks':11*len(rows),'failed_checks':failures,
            'source_intervals_method_match':'NOT reproduced exactly; source CI method is undocumented in inspected files. Paired-normal comparator assumes iid images.',
            'source_interval_max_abs_error_pp':max(interval_diffs),
            'toy_cells':len(trows),'toy_sd_zero_cells':sum(r['source_sd']==0 for r in trows),
            'sae_schema_counts':{k:stats[k] for k in ('judged','a','b','c','eventShareA')},
            'sae_a_count_fraction':stats['a']/stats['judged'],
            'sae_denominator_warning':'Count unit and event-weight construction are not independently reconstructed from the aggregate file.',
            'unavailable':['per-image predictions','logits','image identifiers','class-level transitions','checkpoints','author experiment runners','individual SAE judgements','toy run-level seeds/results'],
            'independence_warning':'345 cells reuse 45000 evaluation images; they are not 345 independent model replications.',
            'frontier':frontier}
    (RES/'source_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    # One chart per figure; disjoint prediction transition categories, with net marker.
    p75=[r for r in selected if r['percentile']==75]
    fig,ax=plt.subplots(figsize=(8.3,4.4))
    y=np.arange(len(p75));left=np.zeros(len(p75))
    for key,label,col in [('damage_pct','Correct → wrong',COL['damage']),('repair_pct','Wrong → correct',COL['repair']),('wrong_to_wrong_pct','Wrong → different wrong',COL['other'])]:
        vals=np.array([r[key] for r in p75]);ax.barh(y,vals,left=left,label=label,color=col,height=.58);left+=vals
    ax.scatter([-r['delta_pp'] for r in p75],y,marker='D',s=43,color=COL['net'],label='Net accuracy loss',zorder=5)
    ax.set_yticks(y,[('Inception v1 (final only)' if r['model']=='inception' else r['title']+' (all blocks)') for r in p75]);ax.invert_yaxis()
    ax.set_xlabel('Percentage of the 45,000 evaluation images');ax.set_title('75th-percentile ablation: net loss and prediction turnover',loc='left',fontsize=12)
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right',fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(FIG/'ablation_turnover.pdf');fig.savefig(FIG/'ablation_turnover.png',dpi=180);plt.close(fig)
    # Detailed medium model sensitivity, avoiding cross-architecture conflation.
    rr=[r for r in selected if r['model']=='medium']
    fig,ax=plt.subplots(figsize=(7.8,4.3))
    for key,label,col,mk in [('disagreement_pct','Prediction disagreement',COL['other'],'o'),('damage_pct','Correct → wrong',COL['damage'],'s'),('repair_pct','Wrong → correct',COL['repair'],'^')]:
        ax.plot([r['percentile'] for r in rr],[r[key] for r in rr],marker=mk,label=label,color=col)
    ax.plot([r['percentile'] for r in rr],[-r['delta_pp'] for r in rr],marker='D',label='Net accuracy loss',color=COL['net'])
    ax.set_xlabel('Per-channel magnitude percentile cut');ax.set_ylabel('Evaluation images (%)');ax.set_xticks([25,50,75,95,99]);ax.set_title('ConvNeXt medium: sensitivity across all-block cutoffs',loc='left',fontsize=12)
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(FIG/'ablation_sensitivity.pdf');fig.savefig(FIG/'ablation_sensitivity.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))
    for r in p75:print({k:r[k] for k in ('model','delta_pp','disagreement_pct','damage_pct','repair_pct','wrong_to_wrong_pct','conditional_damage_pct')})
if __name__=='__main__':main()
