# Prospective supplementary and provenance notes

The original contract, code, models, report maps, decoder coefficients and discovery results are preserved under `freeze.json`. No model/report selection is changed by these notes. This file is written before opening the locked test outcomes.

## Group definition correction

The original contract says “strongest positive Pearson coactivation”. The exact frozen implementation greedily takes the **largest remaining correlation**, not a correlation constrained to be positive, until every unit belongs to a pair. Correlation involving a zero-variance unit is assigned zero in the implementation. One terminal pair in seed 29 has correlation about −0.249; other terminal pairs can include zero-variance units. The report therefore calls this baseline **greedy coactivation-paired sum predicates**, with no claim that every pair is positively correlated. There is no change to any group or result.

## Supplementary design timing

The exact affine-dual audit and precision-certified native-state repair were proposed after seeing the original discovery results, but before reading any locked test outcome. They are supplementary prospective tests rather than part of the original contract. Each has its own frozen parameters and receipts. The original test remains valid for its originally frozen methods; the supplementary methods must be identified as supplementary.

## Dataset provenance and licence

The preserved `digits.csv.gz` is the 1,797-image data file bundled with the locally installed scikit-learn 1.9.0 package. NumPy-only code reads it directly; scikit-learn is not used for fitting. It is the scikit-learn subset of the UCI optical handwritten-digit dataset, not the complete 5,620-image UCI corpus. The official scikit-learn documentation describes 1,797 samples with 64 integer-valued features, and the UCI record describes the original 8×8 preprocessing and distinct-writer source partitions. This experiment re-splits the bundled subset by image; writer identities are unavailable. Its test therefore does not establish writer-independent generalisation.

Dataset credit: E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository, DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49). The [UCI record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) identifies the dataset licence as CC BY 4.0. [scikit-learn load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) describes its bundled subset. Both records were checked on 15 October 2026. We preserve the original bundled compressed data without altering its values.
