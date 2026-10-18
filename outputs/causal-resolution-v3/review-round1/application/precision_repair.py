#!/usr/bin/env python3
"""Weight-only native precision repair with exact global box certificates.

This supplementary design was specified after discovery but before test access.
It is a numerical sufficiency baseline, not a semantic explanation.
"""
from __future__ import annotations
import argparse, json, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import experiment as ex

P=Path(__file__).resolve().parent
EPS=[.25,.5,1.,2.,4.]

def exact_bound(m,c,bits):
    return [sum((Fraction(int(m[j])*abs(int(c[j,k])),ex.H_DEN*10*ex.SCALE*(1<<int(bits[j]))) for j in range(32)),Fraction()) for k in range(10)]

def greedy_bits(m,c,eps):
    bits=np.zeros(32,dtype=np.int64);limit=Fraction(str(eps));trace=[]
    terms=[[Fraction(int(m[j])*abs(int(c[j,k])),ex.H_DEN*10*ex.SCALE) for k in range(10)] for j in range(32)]
    total=[sum((terms[j][k] for j in range(32)),Fraction()) for k in range(10)]
    while max(total)>limit:
        options=[]
        for j in range(32):
            if m[j]==0:continue
            bound=[total[k]-terms[j][k]/2 for k in range(10)]
            options.append((max(bound),sum(bound),j))
        _,_,j=min(options);bits[j]+=1
        terms[j]=[v/2 for v in terms[j]]
        total=[total[k]-terms[j][k] for k in range(10)]
        trace.append(j)
        if len(trace)>1024:raise RuntimeError('Unexpected precision growth')
    assert total==exact_bound(m,c,bits)
    return bits,trace

def freeze():
    if (P/'precision_freeze.json').exists():raise FileExistsError('Already sealed')
    if (P/'test_results.json').exists():raise RuntimeError('Test was already evaluated')
    contract={'designed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'timing':'supplementary design after original discovery results; before any locked test outcomes','primary_epsilon':1.0,'epsilons':EPS,'domain':'all x_i in [0,16], 64 inputs, through exported ReLU model; any coordinate attenuation in [0,1]; all endpoint states inside certified hidden box','hidden_box_bound_integer':'M_j=max(0,16*(sum_i max(W1_int[i,j],0)+b1_int[j]))','code':'floor(h_j*2^b_j/M_j), clipped to [0,2^b_j−1], interval midpoint reconstruction','response_decoder':'original fixed centered linear readout of reconstructed states; coefficient count10 times number of positive-range hidden units','uniform':'minimum common b satisfying exact paired bound for all10 output coordinates','gain_aware':'start b_j=0, increment j that minimizes max exact output bound; tie by sum bounds then lowest index; stop when max bound<=epsilon','certificate':'sum_j M_j*abs(10W2_int[j,k]−sum_l W2_int[j,l])/(65536*10*4096*2^b_j)<=epsilon for every k','midpoint_factor':'single state bound uses half an interval; two endpoints consume a full interval, already included above','claim':'global box upper certificate for finite precision native-state report; not minimal-bit proof or semantic legibility','test':'same frozen test banks and metrics as original experiment; no parameter refitting','cost':'sum_j b_j bits/state; twice that for pair; per-unit uniform arithmetic quantiser; fixed dense linear readout'}
    ex.write_once(P/'precision_contract.json',contract)
    records=[]
    for seed in ex.SEEDS:
        sd=P/f'seed-{seed}';model=np.load(sd/'model.npz')
        m=np.maximum(0,16*(np.maximum(model['w1'],0).sum(axis=0)+model['b1']))
        c=10*model['w2']-model['w2'].sum(axis=1,keepdims=True)
        rec={'seed':seed,'positive_range_units':int((m>0).sum()),'m_integer':m.tolist(),'configs':{}}
        for eps in EPS:
            uniform=0
            while True:
                ub=np.where(m>0,uniform,0).astype(np.int64)
                if max(exact_bound(m,c,ub))<=Fraction(str(eps)):break
                uniform+=1
            gb,trace=greedy_bits(m,c,eps)
            for method,bits in [('uniform',ub),('gain',gb)]:
                name=f'{method}-{eps:g}';bound=exact_bound(m,c,bits)
                np.savez_compressed(sd/f'precision-{name}.npz',M=m,C=c,bits=bits,epsilon=np.asarray(eps))
                rec['configs'][name]={'epsilon':eps,'state_bits':int(bits.sum()),'bits_by_unit':bits.tolist(),'max_bound':float(max(bound)),'bounds_exact':[str(v) for v in bound],'bounds_float':[float(v) for v in bound],'greedy_order':trace if method=='gain' else None,'decoder_coefficients':10*int((m>0).sum())}
        records.append(rec)
    ex.dump(P/'precision_discovery.json',records)
    files={str(p.relative_to(P)):ex.sha(p) for p in sorted(P.glob('seed-*/precision-*.npz'))}
    files.update({f:ex.sha(P/f) for f in ['precision_contract.json','precision_repair.py','precision_discovery.json','freeze.json']})
    ex.write_once(P/'precision_freeze.json',{'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_opened':False,'files':files})
    print(json.dumps({'phase':'precision_sealed','sha256':ex.sha(P/'precision_freeze.json'),'primary':[{'seed':r['seed'],'uniform_bits':r['configs']['uniform-1']['state_bits'],'gain_bits':r['configs']['gain-1']['state_bits'],'gain_bound':r['configs']['gain-1']['max_bound']} for r in records]}),flush=True)

def predict(h4,h4p,o):
    m=o['M'];c=o['C'];b=o['bits'];levels=np.left_shift(1,b);safe=np.where(m>0,4*m,1)
    q=np.minimum((h4*levels)//safe,levels-1);qp=np.minimum((h4p*levels)//safe,levels-1)
    q[:,m==0]=0;qp[:,m==0]=0
    maxlevel=int(levels.max());mult=maxlevel//levels
    predicted_num=((qp-q)*(m*mult))@c
    denominator=ex.H_DEN*ex.SCALE*10*maxlevel
    return predicted_num,denominator,q,qp

def evaluate():
    if (P/'precision_test_results.json').exists():raise FileExistsError('Already tested')
    lock=json.loads((P/'precision_freeze.json').read_text())
    for f,d in lock['files'].items():
        if ex.sha(P/f)!=d:raise ValueError('Freeze mismatch: '+f)
    # Original test phase must have opened the test first, preserving its own receipts.
    if not (P/'test_results.json').exists():raise RuntimeError('Run frozen original test evaluation first')
    te=np.load(P/'test.npz');contract=json.loads((P/'contract.json').read_text());groups=np.asarray(contract['interventions']['groups']);results=[]
    for seed in ex.SEEDS:
        sd=P/f'seed-{seed}';model=np.load(sd/'model.npz');h=ex.hidden(te['x'],model)
        banks={'seen_strength':ex.bank(h,te['ids'],[.5,1.],groups),'unseen_strength':ex.bank(h,te['ids'],[.25,.75],groups),'natural_pairs':ex.natural_bank(h,te['ids'])};rec={'seed':seed,'banks':{}}
        for name,(h4,h4p,meta) in banks.items():
            yn=ex.target_num(h4,h4p,model);r={}
            for path in sorted(sd.glob('precision-*.npz')):
                o=np.load(path);pn,den,q,qp=predict(h4,h4p,o);scale=den//ex.RESPONSE_DEN
                if den%ex.RESPONSE_DEN:raise ValueError('Unexpected denominator relation')
                error_num=pn-yn*scale
                eps=Fraction(str(float(o['epsilon'])))
                # Exact finite verification: no float tolerance used in pass/fail.
                assert int(np.abs(error_num).max())*eps.denominator <= den*eps.numerator
                assert np.all(h4<=4*o['M']) and np.all(h4p<=4*o['M'])
                sr=ex.score(pn/den,yn);sr['exact_max_error_numerator']=int(np.abs(error_num).max());sr['exact_error_denominator']=den;sr['exact_bank_pass']=True;sr['state_bits']=int(o['bits'].sum());sr['certified_max_error']=float(max(exact_bound(o['M'],o['C'],o['bits'])))
                r[path.stem.removeprefix('precision-')]=sr
            rec['banks'][name]=r
        results.append(rec)
    ex.write_once(P/'precision_test_results.json',{'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'precision_freeze_sha256':ex.sha(P/'precision_freeze.json'),'results':results})
    print(json.dumps({'phase':'precision_test_complete','all_exact_checks_passed':True}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);a=p.parse_args();{'freeze':freeze,'evaluate':evaluate}[a.phase]()
