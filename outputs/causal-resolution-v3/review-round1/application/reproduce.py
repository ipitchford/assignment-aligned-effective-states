#!/usr/bin/env python3
"""Regenerate models/reports/evaluations from the preserved exact split files.

Usage: python3 reproduce.py /absolute/path/to/new-empty-directory
No external dataset path, downloads, scipy or sklearn are needed.
"""
from pathlib import Path
import shutil,sys
import experiment as ex
import precision_repair as pr

source=Path(__file__).resolve().parent
if len(sys.argv)!=2:raise SystemExit(__doc__)
out=Path(sys.argv[1]).resolve()
if out.exists() and any(out.iterdir()):raise SystemExit('Destination must be new or empty.')
out.mkdir(parents=True,exist_ok=True)
for name in ['experiment.py','precision_repair.py','verify.py','summarise.py','reproduce.py','contract.json','test_lock.json','train.npz','discovery_fit.npz','discovery_validation.npz','test.npz','digits.csv.gz','pre_test_notes.md']:
    shutil.copyfile(source/name,out/name)
ex.HERE=out;pr.P=out
ex.fit();pr.freeze();ex.evaluate();pr.evaluate()
print('Reproduced fixed-split study in '+str(out))
