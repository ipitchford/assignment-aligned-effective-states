"""Plot the frozen SAT development comparison; no fitting or inference."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sat_margin_baseline" / "results.json"
OUT = ROOT / "paper" / "figures"


def main():
    result = json.loads(SOURCE.read_text())
    primary = result["aggregate"]["primary_assessment"]
    bootstrap = result["bootstrap"]
    margin = result["interpretation"]["resolution"] * 100
    methods = ["structured", "continuous"]
    labels = ["Readiness rule", "Continuous-margin fit"]
    means = np.array([primary[m]["macro"] for m in methods]) * 100
    intervals = np.array([bootstrap[m]["interval_2_5_97_5"] for m in methods]) * 100
    assert np.all(intervals[:, 0] <= means) and np.all(means <= intervals[:, 1])
    delta = primary["delta"]["macro"] * 100
    lo, hi = np.array(bootstrap["delta"]["interval_2_5_97_5"]) * 100
    assert abs(delta - (means[1] - means[0])) < 1e-10

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.65),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    colors = ["#216869", "#a04d23"]
    for i, (mean, ci, color) in enumerate(zip(means, intervals, colors)):
        ax.errorbar(mean, 1 - i, xerr=[[mean-ci[0]], [ci[1]-mean]],
                    fmt="o", color=color, capsize=4, markersize=7, lw=1.8)
        ax.text(mean, 1-i+0.19, f"{mean:.2f}%", ha="center", color=color)
    ax.set(yticks=[1, 0], yticklabels=labels, ylim=(-0.48, 1.55),
           xlim=(85, 95), xlabel="Balanced accuracy (%)",
           title="A  Same native activation target")
    ax.grid(axis="x", color="#e1e1e1", linewidth=.6)
    ax.set_axisbelow(True)

    bx.axvspan(-margin, margin, color="#e9edf0", zorder=0)
    bx.axvline(0, color="#666666", lw=.8)
    bx.axvline(-margin, color="#999999", lw=.8, ls="--")
    bx.axvline(margin, color="#999999", lw=.8, ls="--")
    bx.errorbar(delta, 0, xerr=[[delta-lo], [hi-delta]], fmt="o",
                color="#313c4b", capsize=4, lw=1.8, markersize=7)
    bx.text(delta, .18, f"{delta:.2f} pp", ha="center", color="#313c4b")
    bx.text(0, -.32, "Declared ±2 pp range", ha="center", fontsize=8)
    bx.set(yticks=[], ylim=(-.5, .55), xlim=(-4, 3),
           xlabel="Continuous fit − readiness\n(percentage points)",
           title="B  Paired difference")
    bx.spines["left"].set_visible(False)
    fig.subplots_adjust(left=.22, right=.98, top=.83, bottom=.25, wspace=.36)
    fig.text(.5, .055,
             "95% origin-cluster bootstrap intervals · 685 origins · unequal supervision · development data",
             ha="center", fontsize=8, color="#444444")
    OUT.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"native-sat-comparison.{suffix}", dpi=180,
                    metadata={"Creator": "plot_native_sat.py"})
    plt.close(fig)
    receipt = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "means_percent": dict(zip(methods, means.tolist())),
        "delta_percentage_points": delta,
        "delta_interval_percentage_points": [lo, hi],
        "scope": "Figure from frozen development results; no new fitting, inference or statistical analysis.",
    }
    (OUT / "native-sat-comparison.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
