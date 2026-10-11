# Evaluation

How results become metrics and paired comparisons.

```mermaid
flowchart LR
    csv["results CSV"]
    agg["criterion rows → overall<br/>via criteria_aggregation"]
    hum["overall human score<br/>from the dataset"]
    met["per dataset:<br/>NMAE · NRMSE · QWK ·<br/>Pearson · Spearman<br/>(+ MAE, RMSE, custom)"]
    avg["dataset-equal average row"]
    cmp["compare: paired Wilcoxon<br/>on absolute error, per dataset"]
    csv --> agg --> met
    hum --> met
    met --> avg
    met --> cmp
```

`run` computes metrics automatically when `runtime.compute_metrics` is true. `metrics`,
`compare`, and `baseline` rerun the same code on CSVs already on disk. NMAE, the primary metric,
is `10 × MAE / (score_max − score_min)`, which puts datasets with different scales on one 0–10
reference.

`adjudication` reports how the final-agent procedures resolve a pool. It recomputes every DA
outcome from the pool's source results with the disagreement rule the procedure applies, counts
RA outcomes from the pool's reviews CSVs, and averages the reviews' completion tokens per essay
for each reviewed source. Boolean review columns (`rubric_consistency`, `evidence_grounding`,
`is_admissible`) are `1` or `0` in every CSV the framework writes or reads.

---

[← Final-agent procedures](05-final-agent-procedures.md) · [Overview](README.md) · [Overview](README.md)
