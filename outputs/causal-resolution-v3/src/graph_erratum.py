"""Exact threshold controls and the corrected C5 uncertainty illustration.

Run from any directory. Checks the API's finite-tolerance boundary and retains
the strict/open endpoints rather than interpolating integer graph bounds.
"""
from pathlib import Path
from fractions import Fraction as Q
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from resolution import incompatibility_graph, chromatic_number

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = [[Q(0) for _ in range(5)] for _ in range(5)]
    for j in range(5):
        rows[j][j] = Q(3, 2)
        rows[(j+1) % 5][j] = Q(-3, 2)
    r = np.asarray(rows, dtype=float)
    checks = []
    for delta in map(Q, ['0', '0.249999', '0.25', '0.250001',
                         '0.499999', '0.5', '0.500001', '0.75']):
        bounds = []
        for epsilon in [1+delta, 1-delta]:
            adj = np.array([[max(abs(a-b) for a, b in zip(x, y)) > 2*epsilon
                             for y in rows] for x in rows])
            exact = chromatic_number(adj)[0]
            numeric = chromatic_number(incompatibility_graph(r, float(epsilon)))[0]
            assert exact == numeric
            bounds.append(exact)
        assert bounds == [3 if delta < Q(1, 2) else 1,
                          3 if delta <= Q(1, 4) else 5]
        checks.append({'delta': str(delta), 'lower': bounds[0], 'upper': bounds[1]})
    rejected = 0
    for epsilon in [float('nan'), float('inf'), -float('inf'), -1]:
        try:
            incompatibility_graph(r, epsilon)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('invalid tolerance accepted')
    try:
        incompatibility_graph(np.empty((2, 0)), 1)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError('empty response coordinates accepted')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    low, high = '#1c6d8f', '#b95529'
    ax.plot([0, .5], [3, 3], color=low, lw=2, label='Certified lower count')
    ax.plot([.5, .75], [1, 1], color=low, lw=2)
    ax.plot([0, .25], [3, 3], color=high, lw=2, ls='--', label='Certified upper count')
    ax.plot([.25, .75], [5, 5], color=high, lw=2, ls='--')
    for x, y, colour, filled in [(.5, 3, low, False), (.5, 1, low, True),
                                  (.25, 3, high, True), (.25, 5, high, False)]:
        ax.scatter([x], [y], s=48, facecolor=colour if filled else 'white',
                   edgecolor=colour, zorder=4, linewidth=1.5)
    ax.axhline(3, color='#444444', lw=.7, alpha=.3)
    ax.set(xlim=(-.015, .77), ylim=(.7, 5.35), xticks=[0, .25, .5, .75],
           yticks=[1, 2, 3, 4, 5], ylabel='Number of response labels',
           xlabel=r'Uniform response allowance $\delta$; tolerance $\varepsilon=1$')
    ax.legend(loc='upper left', bbox_to_anchor=(0, .84), frameon=False, fontsize=9)
    fig.tight_layout()
    dest = ROOT/'paper'/'figures'
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(dest/f'c5_corrected.{ext}', dpi=180, bbox_inches='tight')
    result = {'status': 'PASS', 'exact_threshold_cases': checks,
              'invalid_inputs_rejected': rejected,
              'interpretation': 'outer bracket; no sharp identification claim'}
    (ROOT/'data'/'derived'/'graph_erratum_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
