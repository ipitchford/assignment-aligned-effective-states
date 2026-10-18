"""Independent rank-sum AUROC and semantic recount from retained fresh arrays."""
from pathlib import Path
import argparse,hashlib,json,sys,itertools
import numpy as np
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--arrays',default=str(HERE/'observed/native-outputs.npz'));ap.add_argument('--out',required=True);args=ap.parse_args()
def require(ok,message):
    if not ok:raise RuntimeError(message)
with np.load(HERE/'bank/bank.npz',allow_pickle=False) as z:tokens=z['tokens'].copy()
with np.load(args.arrays,allow_pickle=False) as z:pre=z['pre'];post=z['post'];logits=z['logits']
receipt=json.loads((HERE/'observed/receipt.json').read_text())
require(hashlib.sha256((HERE/'bank/bank.npz').read_bytes()).hexdigest()==receipt['bank_sha256'],'bank hash')
for array,shape in [(pre,(4096,512)),(post,(4096,512)),(logits,(4096,16))]:require(array.shape==shape and np.isfinite(array).all(),'shape/finite')
require(np.array_equal(post,np.maximum(pre,0)),'ReLU')
truth=np.empty((4096,32),bool)
for a,bits in enumerate(itertools.product((False,True),repeat=5)):
    lit=np.asarray(bits+tuple(not b for b in bits))
    truth[:,a]=np.logical_or(lit[tokens[:,1:40:4]],lit[tokens[:,2:40:4]]).all(1)
task=np.where(truth.any(1),13,14)
require(np.array_equal(task,tokens[:,41]),'task truth')
group={};nsol=truth.sum(1)
for row in receipt['per_unit']:
    unit,label=row['unit'],row['assignment_id_msb_x0'];y=truth[:,label];n=int(y.sum());m=len(y)-n
    ranks=rankdata(pre[:,unit],method='average');auc=float((ranks[y].sum()-n*(n+1)/2)/(n*m))
    require(abs(auc-row['auroc'])<1e-15,'rank-sum AUROC')
    group.setdefault(label,[]).append(auc);detected=post[:,unit]>=0.5
    for name,mask in [('sole_positive',(nsol==1)&y),('multi_positive',(nsol>=2)&y),('unsat_negative',nsol==0),('sole_negative',(nsol==1)&~y),('multi_negative',(nsol>=2)&~y)]:
        require(row['threshold_cells'][name]=={'detected':int((detected&mask).sum()),'denominator':int(mask.sum())},'threshold cell')
require(float(np.mean([np.mean(v) for v in group.values()]))==receipt['assignment_macro_auroc'],'macro')
require(int((logits.argmax(1)==task).sum())==receipt['task']['correct'],'native accuracy')
record={'status':'PASS','optimize':sys.flags.optimize,'rows':4096,'unit_auroc_checks':17,'threshold_cell_checks':85,'scope':'independent Boolean evaluation, rank-sum AUROC, threshold counts and native accuracy; no new model forward','native_receipt_sha256':hashlib.sha256((HERE/'observed/receipt.json').read_bytes()).hexdigest()}
with Path(args.out).open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
