"""Bind the primary discovery banks to images and frozen model parameters.

This implementation uses Python integer endpoint evaluation and does not import
the study's target-generation or feature-selection routines.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    app = ROOT/'application'
    split = np.load(app/'discovery_fit.npz')
    contract = json.loads((app/'contract.json').read_text())
    groups = contract['interventions']['groups']
    lookup = {int(i): j for j, i in enumerate(split['ids'])}
    checks = 0
    records = []
    for seed in [17, 29, 43]:
        folder = app/f'seed-{seed}'
        m = np.load(folder/'model.npz')
        h = np.maximum(split['x'].astype(object)@m['w1'].astype(object)
                       + 16*m['b1'].astype(object), 0)
        bank = np.load(folder/'discovery_bank.npz')
        meta = bank['meta']
        assert meta.shape == (1904, 3)
        assert set(map(int, meta[:, 2])) == {2, 4}
        before, after = [], []
        for image_id, group, strength in meta:
            hh = 4*h[lookup[int(image_id)]]
            hp = hh.copy()
            hp[groups[int(group)]] = (4-int(strength))*h[lookup[int(image_id)]][groups[int(group)]]
            before.append(hh)
            after.append(hp)
        before, after = np.asarray(before, dtype=object), np.asarray(after, dtype=object)
        outputs = []
        for endpoint in [before, after]:
            z = endpoint@m['w2'].astype(object) + 4*65536*m['b2'].astype(object)
            outputs.append(10*z-z.sum(axis=1, keepdims=True))
        assert int(bank['response_denominator']) == 4*65536*4096*10
        assert np.array_equal(outputs[1]-outputs[0], bank['response_numerator'])
        report = np.load(folder/'report-native-8.npz')
        for weights, thresholds, expected in [
            (report['w'], report['threshold'], bank['A']),
            (bank['candidate_w'], bank['candidate_threshold'], bank['candidate_A'])
        ]:
            first = (before@weights.astype(object).T > 4*thresholds.astype(object)).astype(int)
            last = (after@weights.astype(object).T > 4*thresholds.astype(object)).astype(int)
            assert np.array_equal(last-first, expected)
        checks += 6
        records.append({'seed': seed, 'exact_response_rows': len(meta),
                        'source_bank_sha256': hashlib.sha256((folder/'discovery_bank.npz').read_bytes()).hexdigest()})
    result = {'status': 'PASS', 'checks': checks, 'response_rows': 5712,
              'records': records, 'scope': 'image/model-to-discovery-bank exact arithmetic binding'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
