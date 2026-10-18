#!/usr/bin/env python3
"""Apply release-language and claim-ledger polish to manuscript v4.6."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "outputs/causal-resolution-v3/paper/manuscript.md"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text()
    text = replace_once(
        text,
        'date: "18 October 2026 · Version 4.6 working draft · Unrefereed"',
        'date: "18 October 2026 · Version 4.6 research release · Unrefereed"',
        "release label",
    )
    text = replace_once(
        text,
        "The SAT studies identify the checkpoint's effective final-MLP state as a sparse native assignment report with target-dependent limits.",
        "The SAT studies identify the checkpoint's effective final-MLP state as a graded assignment-aligned report with target-dependent limits.",
        "claim ledger heading",
    )
    text = replace_once(
        text,
        "Within that live span, native assignment matching still exceeds every dimension-matched rotation and positive-mixture reference. Prevalence-preserving shuffled labels and one-violation controls reverse direction, while orbit-grouped probes show that all assignments are linearly present even when three lack equally clean single native axes. This supports unusually legible native axes *within this effective state*, while leaving prevalence, development and capacity dependence unresolved.",
        "Within that live span, native assignment matching exceeds every fixed dimension-matched rotation and positive-mixture reference. Prevalence-preserving shuffled labels reverse direction, the one-violation band does not favour native coordinates, and orbit-grouped probes show that all assignments are linearly present even when three lack equally clean single native axes. Learned 34-component NMF slightly exceeds the native match while recovering 32 native axes above cosine 0.90. This supports an assignment-aligned nonnegative state with strong native resemblance, while leaving prevalence, development and capacity dependence unresolved.",
        "claim ledger learned basis",
    )
    text = replace_once(
        text,
        "| Characterise the complete final-MLP state | Protected full-space comparison; retrospective geometry and probe controls | All 512 coordinates and the bank-effective 34-dimensional state | Native matching exceeds every raw and standardised dimension-matched rotation and positive-mixture reference. Probes recover three assignments lacking equally clean single axes. |",
        "| Characterise the complete final-MLP state | Protected full-space comparison; retrospective geometry, learned-basis and probe controls | All 512 coordinates and the bank-effective 34-dimensional state | Native matching exceeds fixed rotations and mixtures; NMF slightly improves the mean while recovering 32 native axes. Probes recover three assignments lacking equally clean single axes. |",
        "claim ledger table",
    )
    text = replace_once(
        text,
        "| Account for the exact final readout | Retrospective exact arithmetic on retained states | Complete state, zeroing, category margins and cross-fitted threshold substitution | The readout sums positive SAT evidence. All-state removal lowers accuracy to 20.06%; threshold substitution preserves 98.39% of argmaxes and lowers accuracy to 98.15%. |",
        "| Account for the exact final readout | Retrospective exact arithmetic on retained states | Complete state, zeroing, category margins, threshold substitution and refitted readouts | The readout sums positive SAT evidence. All-state removal lowers accuracy to 20.06%; fixed-readout threshold substitution gives 98.15%, while refitted bit and continuous classifiers give 99.12% and 99.68%. |",
        "claim ledger readout",
    )
    PATH.write_text(text)
    print("polished", PATH)


if __name__ == "__main__":
    main()
