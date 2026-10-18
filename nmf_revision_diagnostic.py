"""Attribute the retrospective NMF comparison using retained arrays only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    full = root / 'work/next_phase/sat_full_layer_basis_v7_20261018'
    sys.path.insert(0, str(full))
    from effective_state_followup_v6 import SOURCE_REPORT, assignment_name, two_sided_auc
    from analyze import maximum_matching
    from protocol import BANK, semantic_counts
    native_path = full / 'full_native_outputs.npz'
    with np.load(native_path, allow_pickle=False) as native:
        state = native['post'][:, SOURCE_REPORT].astype(np.float64)
    with np.load(BANK, allow_pickle=False) as bank:
        labels = semantic_counts(bank['unique_clauses']) == 0
    native_axes, native_scores = maximum_matching(two_sided_auc(state, labels))
    active = state > 0
    single = active.sum(axis=1) == 1
    anchors = []
    for column, unit in enumerate(SOURCE_REPORT):
        count = int(np.count_nonzero(single & active[:, column]))
        active_count = int(active[:, column].sum())
        anchors.append({'unit': int(unit), 'single_active_rows': count,
                        'active_rows': active_count, 'fraction_all_rows': count / len(state),
                        'fraction_unit_active_rows': count / active_count})
    expected = json.loads((full / 'effective_state_followup_v6.json').read_text())
    # Discover the retained family without depending on unrelated diagnostic keys.
    def family(value):
        if isinstance(value, dict):
            if 'NMF_k34' in value:
                return value['NMF_k34']['runs']
            for child in value.values():
                found = family(child)
                if found is not None:
                    return found
        return None
    old_runs = family(expected)
    if old_runs is None:
        raise RuntimeError('retained NMF comparison missing')
    runs = []
    for seed in range(5):
        model = NMF(n_components=34, init='nndsvdar', random_state=seed,
                    max_iter=2000, tol=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            scores = model.fit_transform(state)
        selected, values = maximum_matching(two_sided_auc(scores, labels))
        directions = model.components_ / np.linalg.norm(model.components_, axis=1)[:, None]
        component_axes, cosines = maximum_matching(np.abs(directions))
        delta = values - native_scores
        rows = [{'assignment': assignment_name(a), 'native_unit': int(SOURCE_REPORT[native_axes[a]]),
                 'native_auc': float(native_scores[a]), 'nmf_component': int(selected[a]),
                 'nmf_auc': float(values[a]), 'delta': float(delta[a]),
                 'nmf_matched_native_unit': int(SOURCE_REPORT[component_axes[selected[a]]]),
                 'component_axis_cosine': float(cosines[selected[a]])} for a in range(32)]
        error = float(np.linalg.norm(state - scores @ model.components_) / np.linalg.norm(state))
        old = old_runs[seed]
        if not np.isclose(values.mean(), old['assignment_matching']['matching_32_mean'], rtol=1e-12, atol=1e-12):
            raise RuntimeError('NMF assignment scores differ from retained v6')
        if not np.isclose(error, old['relative_frobenius_reconstruction_error'], rtol=1e-12, atol=1e-12):
            raise RuntimeError('NMF reconstruction differs from retained v6')
        runs.append({'seed': seed, 'iterations': int(model.n_iter_),
                     'convergence_warnings': [str(w.message) for w in caught],
                     'relative_frobenius_error': error, 'matching_mean': float(values.mean()),
                     'mean_gain': float(delta.mean()), 'per_assignment': rows,
                     'components_at_least_0_99': int((cosines >= .99).sum()),
                     'unrecovered_native_axes': [int(SOURCE_REPORT[component_axes[j]]) for j in range(34) if cosines[j] < .99],
                     'minimum_matched_cosine': float(cosines.min()),
                     'gain_from_components_below_0_99': float(sum(row['delta'] for row in rows if row['component_axis_cosine'] < .99) / 32),
                     'gain_from_components_at_least_0_99': float(sum(row['delta'] for row in rows if row['component_axis_cosine'] >= .99) / 32)})
    result = {'status': 'PASS', 'scope': 'retrospective retained-array diagnostic; zero neural forwards',
              'native_file': str(native_path.relative_to(root)),
              'native_sha256': hashlib.sha256(native_path.read_bytes()).hexdigest(),
              'rows': len(state), 'dimension': state.shape[1],
              'identity_factorization_relative_error': float(np.linalg.norm(state - state @ np.eye(34)) / np.linalg.norm(state)),
              'single_active_rows': int(single.sum()), 'per_unit_anchor_rows': anchors,
              'units_with_single_active_examples': int(sum(row['single_active_rows'] > 0 for row in anchors)),
              'native_matching_mean': float(native_scores.mean()), 'runs': runs}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'anchor_units': result['units_with_single_active_examples'],
                      'single_active_rows': result['single_active_rows'],
                      'runs': [{k: v for k, v in row.items() if k != 'per_assignment'} for row in runs]}, indent=2))


if __name__ == '__main__':
    main()
