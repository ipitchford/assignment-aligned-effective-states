"""Frozen grammar and exact semantic oracle; no neural-model access."""
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CONTRACT_SHA256 = "36a8f2763d4e299d19620db4e58a30e19788ae404f56939225f73cf95c489104"
PUBLISHED_UNITS = np.array([10,29,36,41,48,55,77,81,96,150,185,189,195,201,224,231,245,261,291,304,317,326,334,374,380,411,416,435,450,482,490,492,495,499], dtype=np.int64)
CLAUSES = np.array([(i+5*a,j+5*b) for i in range(5) for j in range(i+1,5) for a in range(2) for b in range(2)], dtype=np.int16)
ASSIGNMENTS = ((np.arange(32)[:,None] >> np.arange(4,-1,-1)) & 1).astype(bool)
LITERAL_TRUTH = np.concatenate((ASSIGNMENTS, ~ASSIGNMENTS), axis=1).T
CLAUSE_TRUTH = LITERAL_TRUTH[CLAUSES].any(axis=1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_contract():
    if sha256(ROOT / "contract.md") != CONTRACT_SHA256:
        raise ValueError("Contract changed: version a new contract rather than overwrite outcomes")


def encode(clause_ids):
    clauses = np.asarray(clause_ids)
    if clauses.ndim != 2 or clauses.shape[1] != 10 or np.any((clauses<0)|(clauses>=40)):
        raise ValueError("Invalid canonical formula shape/IDs")
    tokens = np.empty((len(clauses),41), dtype=np.int64)
    tokens[:,0:40:4] = 10
    tokens[:,3:40:4] = 11
    tokens[:,1:40:4] = CLAUSES[clauses,0]
    tokens[:,2:40:4] = CLAUSES[clauses,1]
    tokens[:,40] = 12
    return tokens


def semantic_counts(clause_ids):
    return (~CLAUSE_TRUTH[np.asarray(clause_ids)]).sum(axis=1, dtype=np.int16)


def all_edit_labels(clause_ids, counts=None):
    clauses = np.asarray(clause_ids)
    counts = semantic_counts(clauses) if counts is None else counts
    remainder = counts[:,None,:] - (~CLAUSE_TRUTH[clauses]).astype(np.int16)
    edited_counts = remainder[:,:,None,:] + (~CLAUSE_TRUTH)[None,None,:,:]
    return (edited_counts == 0).any(axis=-1)


def python_truth_table(clause_ids):
    """Independent Boolean implementation without vectorized count updates."""
    table=[]
    for assignment in range(32):
        values=[bool((assignment>>(4-j))&1) for j in range(5)]
        def literal(token):
            return values[token] if token<5 else not values[token-5]
        table.append(all(literal(int(CLAUSES[c,0])) or literal(int(CLAUSES[c,1])) for c in clause_ids))
    return table


def report_codes(bits):
    return np.sum(bits.astype(np.uint64) * (np.uint64(1) << np.arange(34,dtype=np.uint64)),axis=1,dtype=np.uint64)


def correct_margin(logits, labels):
    logits=np.asarray(logits)
    labels=np.asarray(labels)
    other=logits.copy()
    other[np.arange(len(other)),labels]=-np.inf
    return logits[np.arange(len(logits)),labels]-other.max(axis=1)


def clause_string(cid):
    names=[f"x{i}" for i in range(5)]+[f"~x{i}" for i in range(5)]
    return "("+" OR ".join(names[int(i)] for i in CLAUSES[cid])+")"
