# Optional recovery of omitted source text

The portable numerical audit does not require the two complete source documents.
Their exact URLs, original byte lengths and SHA-256 hashes are in
`SOURCE_OMISSIONS.json`. The unchanged original `manifest.json` and
`input_freeze.json` still bind these omitted paths; they are historical full-audit
manifests, not claims that all their files are redistributed here.

To inspect the full source prose, independently obtain the two listed URLs from
the publisher and save them at their listed `inputs/` paths in a separate working
copy. Check both byte length and SHA-256 against `SOURCE_OMISSIONS.json` before
treating them as the exact audited versions. A changed or unavailable URL cannot
be substituted silently. No retrieval runs automatically, and no permission to
redistribute the full text is inferred from public availability.

With exact source bytes recovered, the archival `audit_exports.py --verify` can
recalculate the original report reconciliation in a separate working copy. The
independent checker's full mode additionally requires the original workspace's
prior acquisition paths and earlier seven-file corpus. Those external historical
files are not duplicated here: recovering these two texts alone does not make
that full mode self-contained. The archived full-review receipt identifies its
original scope; the default portable command checks the numerical scope only.

Source factual table values retained in `audit.json` are not a replacement copy
of the complete reports. The numerical replay does not authenticate recorded HTTP
metadata, independently inspect the original seven-file corpus, or verify source
prose identity. It does bind the numerical inputs and recompute the retained
onset, score, denominator and comparison arithmetic.
