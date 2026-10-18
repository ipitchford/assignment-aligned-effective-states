#!/usr/bin/env python3
"""Synchronise v4.6 reader-facing documents from the authoritative manuscript."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/causal-resolution-v3"
MANUSCRIPT = BASE / "paper/manuscript.md"
ABSTRACTS = BASE / "paper/abstracts.md"
README = BASE / "README.md"
STATUS = BASE / "RESEARCH_STATUS.md"


def manuscript_abstract(source: str) -> str:
    start = source.index("abstract: |\n") + len("abstract: |\n")
    end = source.index("\n---\n", start)
    lines = source[start:end].splitlines()
    if not all(line.startswith("  ") for line in lines if line):
        raise RuntimeError("unexpected abstract indentation")
    return "\n".join(line[2:] if line else line for line in lines).strip()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    source = MANUSCRIPT.read_text()
    if "Version 4.6 research release" not in source:
        raise RuntimeError("authoritative manuscript is not the v4.6 release")
    english = manuscript_abstract(source)
    chinese = """Cammarata在《Neural Networks Are Already Clean》中提出：即使較廣的表徵看似分散，原生神經元仍可能是具結構的基本組件。本研究檢驗一個五變數2-SAT轉換器末層MLP的已發表解釋。受保護的7,252條公式資料庫顯示，17個來源指定的單一賦值標籤具有很強的排序能力（宏平均AUROC 0.991；各單元0.961至1.000），但凍結的綜合準則與閾值忠實度準則均未達成。另一項受保護的全層比較把原生賦值配對置於63個固定Haar旋轉之上，但原生稀疏非負座標與稠密旋轉並不可交換。其後的結構分析均為事後分析。來源指定的34個單元構成此資料庫上的完整有效狀態；各單元最佳賦值AUROC為0.946至1.000，活化質量純度為0.595至0.994。線性探針可恢復全部32個賦值。原生配對優於固定旋轉與非負混合，但34成分非負矩陣分解的平均配對略高於原生基底（0.986對0.984），並以大於0.90的餘弦相似度恢復34條原生軸中的32條。因此，此狀態具有強烈的賦值對齊及自然非負幾何，但原生座標並非唯一最佳。最終讀出加總正向SAT證據。全部歸零後準確率為20.06%，接近20.04%的UNSAT比例。固定讀出的閾值替換準確率為98.15%；重新擬合的位元分類器為99.12%，連續活化分類器為99.68%。這個容量充足的單一情境同時符合Already Clean與疊加理論，不能區分兩者。本研究的貢獻是把標籤排序、閾值忠實度、有效狀態幾何、學習所得替代基底、因果使用及行動充分性分開檢驗，並明確指出所需的種子、寬度及上游路徑實驗。"""
    abstract_doc = f"""# Abstracts and keywords

The authoritative English abstract is the abstract field in `manuscript.md`; it is reproduced below for editorial convenience. The Traditional Chinese abstract covers the same evidence and limits without creating a separate claim.

## English abstract

{english}

**English keywords:** mechanistic interpretability; interpretation validation; native neurons; SAT transformers; effective width; causal readout.

## 繁體中文摘要

{chinese}

**關鍵詞：** 機制可解釋性；解釋驗證；原生神經元；SAT轉換器；有效寬度；因果讀出。
"""
    ABSTRACTS.write_text(abstract_doc)

    readme = README.read_text()
    start = readme.index("Version 4.5 identifies")
    end = readme.index("\n\n## What is established", start)
    replacement = """Version 4.6 identifies a graded assignment-aligned effective state in one five-variable 2-SAT transformer. Protected label and complete-layer tests are followed by explicitly retrospective purity, learned-basis, probe, error-localisation and readout analyses. Learned 34-component NMF slightly exceeds native assignment matching while recovering 32 native axes, so the paper supports strong native resemblance without unique native optimality. Exact readout accounting separates fixed-readout threshold substitution (98.15% accuracy) from refitted bit and continuous classifiers (99.12% and 99.68%). The case is compatible with both Already Clean and superposition accounts. See [the research status](RESEARCH_STATUS.md) for completed findings and open scientific questions. Model-generated assessments remain separate provenance records."""
    readme = readme[:start] + replacement + readme[end:]
    readme = replace_once(
        readme,
        "- Within that live state, native matching exceeds every 34-dimensional rotation, 70-column rotation and dense or two-sparse nonnegative-mixture reference. Prevalence-preserving shuffled labels and one-violation controls reverse direction. Orbit-grouped linear probes recover all assignments, including three with weak single native axes.",
        "- Within that live state, native matching exceeds every fixed rotation and nonnegative-mixture reference. Learned 34-component NMF slightly improves the matching mean while recovering 32 native axes above cosine 0.90. Orbit-grouped probes recover all assignments, including three with weak single native axes.",
        "README learned basis bullet",
    )
    readme = replace_once(
        readme,
        "- The final readout is exactly additive. All 34 live units have positive SAT-minus-UNSAT coefficients; grouped zeroing lowers accuracy to 20.06% because redundant positive contributions jointly cross argmax margins. Sub-threshold and rule-inconsistent activations can support correct predictions. This is an exact final-cut result, not an upstream-route result.",
        "- The final readout is exactly additive. All 34 live units have positive SAT-minus-UNSAT coefficients; grouped zeroing lowers accuracy to 20.06%. Fixed-readout threshold substitution gives 98.15% accuracy; orbit-grouped refitted bit and continuous classifiers give 99.12% and 99.68%. This is a final-cut and predictive result, not an upstream-route result.",
        "README readout bullet",
    )
    readme = readme.replace("Version 4.5", "Version 4.6")
    README.write_text(readme)

    status = STATUS.read_text()
    first_end = status.index("\n\n## What the SAT case now establishes")
    status_intro = """# Research status — 18 October 2026

**Version 4.6 is an unrefereed research release.** It centres the paper on one five-variable 2-SAT transformer's assignment-aligned effective final-MLP state. Protected all-seventeen label and complete-layer comparisons are followed by explicitly retrospective purity, learned-basis, probe, error-localisation and refitted-readout controls. Completed observations, computational checks and unresolved scientific questions are distinguished below."""
    status = status_intro + status[first_end:]
    old_retro = """Retrospective analysis shows that Palumbo et al.'s source-nominated 34 units
contain effectively all post-ReLU energy; the other 36 nonzero columns contribute
only 6.98e-53 of squared norm. The rank is 34 across relative tolerances from
1e-12 through 1e-4. Within the exact 34-dimensional live span, native all-32
matching is 0.984 versus a best of 0.915 across 63 rotations. Native also exceeds
all dense and two-sparse nonnegative-mixture references. Prevalence-preserving
shuffled labels and one-violation labels reverse direction. An orbit-grouped
linear probe reaches mean AUROC 0.996 and minimum 0.984, including the three
assignments with weak single native axes. These analyses were designed after the
protected result and do not change its decision."""
    new_retro = """Retrospective analysis shows that Palumbo et al.'s source-nominated 34 units
contain effectively all post-ReLU energy; the other 36 nonzero columns contribute
only 6.98e-53 of squared norm. The rank is 34 across relative tolerances from
1e-12 through 1e-4. Native all-32 matching is 0.984 and exceeds every fixed
rotation and positive-mixture reference. Activation-mass purity nevertheless
ranges from 0.595 to 0.994. Across five seeds, 34-component NMF reaches mean
matching 0.9862--0.9865 and recovers 32 native axes above cosine 0.90. An
orbit-grouped linear probe reaches mean AUROC 0.996 and minimum 0.984. These
analyses were designed after the protected result and do not change its decision;
they support assignment alignment with native resemblance, not unique native
optimality."""
    status = replace_once(status, old_retro, new_retro, "status learned basis")
    old_readout = """Source-threshold misses and false positives can still support
correct decisions. Cross-fitted threshold substitution preserves 98.39% of native
argmaxes while lowering task accuracy from 99.49% to 98.15%, directly measuring
graded magnitude use. These results do not
identify an upstream natural route and zero interventions can be off-manifold."""
    new_readout = """Source-threshold misses and false positives can still support
correct decisions. Fixed-readout threshold substitution preserves 98.39% of native
argmaxes and gives 98.15% task accuracy. Orbit-grouped refitted bit and continuous
classifiers reach 99.12% and 99.68%, separating calibration loss from the remaining
predictive gap. These results do not identify an upstream natural route and zero
interventions can be off-manifold."""
    status = replace_once(status, old_readout, new_readout, "status refitted readout")
    status = status.replace("Version 4.5", "Version 4.6")
    STATUS.write_text(status)
    print("synchronised", ABSTRACTS, README, STATUS)


if __name__ == "__main__":
    main()
