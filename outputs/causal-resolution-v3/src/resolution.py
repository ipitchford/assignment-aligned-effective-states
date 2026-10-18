"""Exact finite response-resolution tools. Suitable for small audit/control cases.

The graph method is a tolerance-aware specialisation of characteristic-graph
functional compression, not a new graph-colouring theorem. Returned labels do
not by themselves carry semantic meaning or imply a small executable encoder.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

def incompatibility_graph(responses: np.ndarray, epsilon: float) -> np.ndarray:
    r=np.asarray(responses,dtype=float)
    if r.ndim!=2 or len(r)==0 or r.shape[1]==0 or not np.isfinite(r).all():
        raise ValueError('responses must be a finite, nonempty states-by-probes matrix')
    if not np.isfinite(epsilon) or epsilon<0:raise ValueError('epsilon must be finite and nonnegative')
    d=np.max(np.abs(r[:,None,:]-r[None,:,:]),axis=2)
    adj=d>2*epsilon
    np.fill_diagonal(adj,False)
    return adj

def chromatic_number(adj: np.ndarray) -> tuple[int,list[int]]:
    a=np.asarray(adj,dtype=bool); n=len(a)
    if a.shape!=(n,n) or not np.array_equal(a,a.T) or np.any(np.diag(a)):
        raise ValueError('Expected simple undirected adjacency matrix')
    if n==0:return 0,[]
    # Exact fixed degree order backtracking; deliberately only small finite cases.
    order=sorted(range(n),key=lambda i:-int(a[i].sum()))
    for k in range(1,n+1):
        colors=[-1]*n
        def visit(pos: int,maxused: int) -> bool:
            if pos==n:return True
            v=order[pos];forbidden={colors[u] for u in range(n) if a[v,u] and colors[u]>=0}
            # Fix first-use symmetry without excluding any partition.
            for c in range(min(k,maxused+2)):
                if c not in forbidden:
                    colors[v]=c
                    if visit(pos+1,max(maxused,c)):return True
            colors[v]=-1
            return False
        if visit(0,-1):return k,colors
    raise AssertionError('Every n-vertex graph is n-colourable')

def midpoint_decoder(responses: np.ndarray, colors: list[int]) -> np.ndarray:
    r=np.asarray(responses,float);out=np.zeros_like(r)
    for c in set(colors):
        mask=np.asarray(colors)==c
        out[mask]=(r[mask].min(0)+r[mask].max(0))/2
    return out

def refinement_number(responses: np.ndarray,native_labels: list[int],epsilon: float) -> int:
    r=np.asarray(responses,float); lab=np.asarray(native_labels)
    if len(lab)!=len(r):raise ValueError('one native label per state is required')
    return max(chromatic_number(incompatibility_graph(r[lab==label],epsilon))[0] for label in set(native_labels))

def exact_bayes_mse(states: np.ndarray,weights: np.ndarray,code: np.ndarray) -> tuple[float,np.ndarray]:
    """Finite conditional-mean risk, sum over target coordinates (not mean per coordinate)."""
    y=np.asarray(states,float);w=np.asarray(weights,float);z=np.asarray(code)
    if len(y)!=len(w) or len(y)!=len(z) or (w<0).any() or not np.isclose(w.sum(),1):
        raise ValueError('Invalid weighted finite distribution')
    groups=defaultdict(list)
    for i,k in enumerate(z):groups[tuple(np.atleast_1d(k).tolist())].append(i)
    pred=np.zeros_like(y)
    for ids in groups.values():
        mass=w[ids].sum()
        if mass>0:pred[ids]=np.sum(w[ids,None]*y[ids],axis=0)/mass
    return float(np.sum(w[:,None]*(y-pred)**2)),pred

def context_incompatibility_graph(responses: np.ndarray,support: np.ndarray,epsilon: float) -> np.ndarray:
    """Encoder sees state only; decoder also sees context.

    responses has shape (states, contexts, probe_coordinates). False support
    denotes a KNOWN inadmissible pair, never a missing measurement. Every
    supported response must be finite. Conflicts are joined across contexts.
    """
    r=np.asarray(responses,float);s=np.asarray(support,bool)
    if r.ndim!=3 or s.shape!=r.shape[:2] or len(r)==0 or r.shape[2]==0 or not np.isfinite(epsilon) or epsilon<0:
        raise ValueError('Invalid context response/support shapes or tolerance')
    if not np.isfinite(r[s]).all():
        raise ValueError('Supported but unmeasured response: cannot certify compatibility')
    a=np.zeros((len(r),len(r)),bool)
    for c in range(r.shape[1]):
        ids=np.flatnonzero(s[:,c])
        if len(ids):
            a[np.ix_(ids,ids)] |= incompatibility_graph(r[ids,c],epsilon)
    return a
