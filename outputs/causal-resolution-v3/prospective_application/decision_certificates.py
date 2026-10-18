#!/usr/bin/env python3
"""Exact decision-regret certificates from existing state reports.

Retrospective descriptive analysis. No models, encoders or decoders are fit.
"""
from pathlib import Path
from fractions import Fraction
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(v,d):return str(Fraction(int(v),int(d)))

def main():
    summary=json.loads((P/'results.json').read_text());records=[]
    for r in summary['results']:
        seed=r['seed'];sd=P/f'seed-{seed}';b=np.load(sd/'bank-unseen_strength.npz');sig=b['signature_second_strength'].reshape(-1,4,10);den=int(b['response_denominator']);z=b['current_output_numerator'];order=np.argsort(-z,axis=1,kind='stable');top=order[:,0];runner=order[:,1];n=len(top)
        drop=np.array([[int(sig[i,g,runner[i]])-int(sig[i,g,top[i]]) for g in range(4)] for i in range(n)],dtype=object);best=drop.max(axis=1);regret=best[:,None]-drop;bestgroup=np.argmax(drop,axis=1)
        for name in ['action-only','native-8','native-12','pca-8','random-8','current-output']:
            q=np.load(sd/f'predictions-unseen_strength-{name}.npz')['state_codes'];signature=np.column_stack([q,top,runner]);_,inv=np.unique(signature,axis=0,return_inverse=True);avg_n=0;worst_n=0;worst_fibre=[];worst_action=None;best_pair=None
            for f in np.unique(inv):
                ix=np.flatnonzero(inv==f);matrix=regret[ix];sums=matrix.sum(axis=0);mx=matrix.max(axis=0);avg_n+=int(sums.min());rho=int(mx.min())
                if rho>worst_n:worst_n=rho;worst_fibre=ix.tolist();worst_action=int(np.argmin(mx))
                for ai,i in enumerate(ix):
                    for j in ix[ai+1:]:
                        if int(bestgroup[i])==int(bestgroup[j]):continue
                        pair_regret=np.maximum(regret[i],regret[j]);rho_pair=int(pair_regret.min());candidate=(rho_pair,-int(i),-int(j))
                        if best_pair is None or candidate>best_pair[0]:best_pair=(candidate,int(i),int(j))
            witness=None
            if best_pair:
                _,i,j=best_pair;reg=np.array([regret[i],regret[j]],dtype=object);pairworst=reg.max(axis=0);rho=int(pairworst.min())
                witness={'rows':[i,j],'input_ids':[int(b['ids'][i]),int(b['ids'][j])],'digit_labels':[int(b['labels'][i]),int(b['labels'][j])],'same_report_code':q[i].tolist(),'same_top_class':int(top[i]),'same_runner_class':int(runner[i]),'best_groups':[int(bestgroup[i]),int(bestgroup[j])],'true_margin_drop_numerators':[[int(v) for v in drop[k]] for k in [i,j]],'true_margin_drops':[[float(Fraction(int(v),den)) for v in drop[k]] for k in [i,j]],'action_regret_numerators':[[int(v) for v in regret[k]] for k in [i,j]],'max_regret_for_each_common_action_numerators':[int(v) for v in pairworst],'any_common_action_worst_regret_exact':exact(rho,den),'any_common_action_worst_regret':float(Fraction(rho,den)),'minimax_common_action':int(np.argmin(pairworst)),'response_denominator':den,'verified_equal_reports_and_context':bool(np.array_equal(q[i],q[j]) and top[i]==top[j] and runner[i]==runner[j]),'both_best_actions_unique':bool(np.sum(drop[i]==best[i])==1 and np.sum(drop[j]==best[j])==1)}
                assert witness['verified_equal_reports_and_context'] and rho>0
            records.append({'seed':seed,'architecture':r['architecture'],'method':name,'policy_input':'state report plus ordered top/runner class labels; no exact logits or margins','strength':.75,'finite_states':n,'context_bits':7,'distinct_report_contexts':len(np.unique(inv)),'empirical_minimum_mean_regret_exact':exact(avg_n,n*den),'empirical_minimum_mean_regret':float(Fraction(avg_n,n*den)),'minimum_worst_case_regret_exact':exact(worst_n,den),'minimum_worst_case_regret':float(Fraction(worst_n,den)),'worst_fibre_rows':worst_fibre,'minimax_action_for_worst_fibre':worst_action,'two_state_witness':witness})
    out={'status':'exact finite decision certificates','scope':'post-test exploratory descriptive study; uniform empirical image weighting; deterministic policy action set has four groups at strength3/4; no population or semantic claim','policy_class':'deterministic action choices; the mean-regret floor also applies to randomised choices by linearity, but the stated worst-case floor is not the randomised minimax value','criterion':'regret in decrease of fixed baseline top-class-versus-runner-up margin; class labels supplied as context but actual margin values are not','formulas':{'per_state_regret':'R_ig=max_a drop_ia−drop_ig','empirical_best_policy_mean':'sum_f min_g sum_(i in f) R_ig / n','best_policy_worst_case':'max_f min_g max_(i in f) R_ig'},'reports_and_class_context_are_frozen':'source encoders retained; this analysis fits no policy','input_sha256':sha(P/'results.json'),'script_sha256':sha(__file__),'records':records}
    (P/'decision_certificates.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps([{'seed':r['seed'],'method':r['method'],'worst_regret_floor':r['minimum_worst_case_regret'],'mean_regret_floor':r['empirical_minimum_mean_regret'],'pair':r['two_state_witness']['input_ids'] if r['two_state_witness'] else None,'pair_floor':r['two_state_witness']['any_common_action_worst_regret'] if r['two_state_witness'] else None} for r in records if r['method']=='native-8']))

if __name__=='__main__':main()
