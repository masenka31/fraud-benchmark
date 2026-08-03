# Retired ablation cells

Records for (dataset, feature set, label regime) cells that are no longer in
`fraud_benchmark.ablation.grid`. Kept as the evidence behind the decision to drop
them; `summarize()` globs `*.jsonl` non-recursively, so nothing here reaches
`results/summary.md`.

`ibm_ccf_subsample_fast` and `ibm_ccf_subsample_slow` (retired 2026-08-03) existed
to give the label-delay axis something to bite on. Their two delay regimes scored
identically — 0.975 against 0.975 at a 0.001 seed noise floor — and their val and
test frauds were 100% `Merchant State == "Italy"`, so they were unusable for the
leakage axis too. `sparkov_slow` carries that role now. See
the retired docs/verification-notes.md, "## The IBM CCF subsamples are retired".
