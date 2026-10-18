# Replay environments

Each `*-installed.lock.txt` pins the complete installed-package snapshot used for
that interpreter; `stage_map.json` assigns every stage. These are environment
snapshots, not wheel hashes or a tested cross-platform installer. The analysis
environment includes unrelated pre-existing packages; their presence is not a
claim that they are all necessary. Use the Python versions in the runtime inventory.
On this host `/opt/homebrew/bin/python3` resolves to the recorded Python 3.14.7.
Native execution and readout-accounting v4 share Python 3.11.14, NumPy 1.26.4 and
Torch 2.1.2. Corrected v6, NMF attribution and the listed scientific checks use
the separate Python 3.11.14/NumPy 2.4.6/SciPy 1.17.1/scikit-learn 1.9.0 runtime.
Preserve a virtual environment's lexical interpreter path when launching it.
The launcher disables Python optimisation so assertion checks remain enabled.
