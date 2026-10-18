#!/usr/bin/env python3
"""Synchronise v4.5 reader-facing documents from the authoritative manuscript."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "outputs/causal-resolution-v3/paper/manuscript.md"
ABSTRACTS = ROOT / "outputs/causal-resolution-v3/paper/abstracts.md"
README = ROOT / "outputs/causal-resolution-v3/README.md"
STATUS = ROOT / "outputs/causal-resolution-v3/RESEARCH_STATUS.md"


def manuscript_abstract(source: str) -> str:
    start = source.index("abstract: |\n") + len("abstract: |\n")
    end = source.index("---\n\n# Introduction", start)
    lines = source[start:end].rstrip().splitlines()
    if not lines or any(not line.startswith("  ") for line in lines):
        raise RuntimeError("unexpected abstract indentation")
    return "\n".join(line[2:] for line in lines)


def main() -> None:
    source = MANUSCRIPT.read_text()
    if "Version 4.5 working draft" not in source:
        raise RuntimeError("authoritative manuscript is not v4.5")
    english = manuscript_abstract(source)
    chinese = (
        "Cammarata在《Neural Networks Are Already Clean》中提出：即使較廣的表徵看似分散，原生神經元仍可能是具結構的基本組件。"
        "本研究稽核一個五變數2-SAT轉換器的末層MLP：名義寬度為512，但在受保護的7,252條公式資料庫上，ReLU後的有效狀態秩為34，而任務只有32個賦值原子。"
        "來源為其中17個單元發表單一賦值標籤。這些標籤具有很強的排序能力（宏平均AUROC 0.991；各單元0.961至1.000），但凍結的綜合準則與閾值忠實度準則均未達成：兩個單元各以32例中的一例超出特異度上限，四個單元未通過一個正例分層。"
        "事後核算顯示，來源指定的34個單元實際包含全部資料庫能量；每個單元都以AUROC 0.946至1.000最佳配對一個賦值，共覆蓋29個不同賦值。依公式軌道分組的線性探針以AUROC 0.986至0.992恢復其餘三個賦值。"
        "在原始、標準化與非負混合比較中，原生一對一配對仍高於63個同維旋轉（原生平均0.984；最佳標準化旋轉0.914），而結構化控制標籤則反向。"
        "最終讀出精確加總34個單元的非負SAT證據。以交叉擬合的閾值位元取代連續活化，保留98.39%的argmax，但使任務準確率由99.49%降至98.15%，直接量出與決策相關的幅度損失。"
        "共同歸零使準確率降至20.06%，證明此末層MLP是該切點唯一的SAT證據通道，而非讀出使用了賦值身分。"
        "因此，名義寬度高估了此檢查點的運作容量。本結果是一個精確的單一檢查點案例：『已經乾淨』的描述大致成立，但仍有未通過的準則、三個共享軸例外，以及在推廣至普遍性、發展來源或較大型對齊系統前必須完成的種子與寬度測試。"
    )
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

    readme = README.read_text().replace("Version 4.4", "Version 4.5")
    old = "Exact readout accounting shows additive positive SAT evidence whose Boolean thresholding loses useful magnitude."
    new = "Exact readout accounting shows additive positive SAT evidence; cross-fitted threshold substitution preserves 98.39% of native argmaxes while lowering task accuracy from 99.49% to 98.15%."
    if old in readme:
        readme = readme.replace(old, new)
    README.write_text(readme)

    status = STATUS.read_text().replace("**Version 4.4 is", "**Version 4.5 is")
    marker = "These analyses were designed after the protected result and do not change its decision."
    addition = marker + " A standardised-coordinate sensitivity gives native 0.984 versus a best rotation of 0.914. Every one of the 34 units is best matched to one assignment at AUROC 0.946--1.000, collectively covering 29 assignments."
    if marker in status and "standardised-coordinate sensitivity" not in status:
        status = status.replace(marker, addition)
    old2 = "Source-threshold misses and false positives can still support\ncorrect decisions, establishing graded magnitude use."
    new2 = "Source-threshold misses and false positives can still support\ncorrect decisions. Cross-fitted threshold substitution preserves 98.39% of native\nargmaxes while lowering task accuracy from 99.49% to 98.15%, directly measuring\ngraded magnitude use."
    if old2 in status:
        status = status.replace(old2, new2)
    STATUS.write_text(status)
    print("synchronised", ABSTRACTS, README, STATUS)


if __name__ == "__main__":
    main()
