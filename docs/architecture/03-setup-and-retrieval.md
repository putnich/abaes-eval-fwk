# Samples, calibration, and retrieval

Which essays are graded, which become few-shot examples, and how examples are picked for each essay.

## Samples and calibration pools

Every dataset an experiment uses becomes one *setup*: a rubric, an evaluation sample, and a
calibration pool to draw few-shot examples from.

```mermaid
flowchart TB
    raw["dataset source<br/>built-in raw file · normalized CSV · loader"]
    frame["validated frame<br/>essay_id, set_id, essay, human_score,<br/>+ criterion columns"]
    rubric["rubric YAML"]
    cal["calibration pool<br/>calibration_strategy (default: stratified)<br/>≤ calibration.max_per_score per score level"]
    rest["remaining essays"]
    sample["evaluation sample<br/>evaluation_strategy (default: quantile_rank)<br/>sample_size per set, or explicit essay_ids"]
    setup["ExperimentalSetup"]

    raw --> frame
    rubric --> frame
    frame --> cal
    frame --> rest
    cal -. excluded from .-> rest
    rest --> sample
    cal --> setup
    sample --> setup
    rubric --> setup
```

Calibration essays are never graded. Pools are cached under `<output_dir>/calibration/`, keyed
by a fingerprint of the data and strategy. A cached pool that no longer matches what the strategy
would produce raises an error (delete it to rebuild) instead of being reused. `prepare` runs this
stage alone and writes the IDs to `<output_dir>/splits/<hash>.json`, with no model or embedding
calls.

A final-agent or review experiment doesn't sample. Its essays are the ones in the prior results
it reuses.

## Example retrieval

Few-shot examples come from the calibration pool, stratified by human score so the model sees
the scale's range rather than near-duplicates of one level.

```mermaid
flowchart TB
    start(["submission + criterion"])
    zs{"zero_shot, or<br/>n_ex_per_score = 0, or<br/>max_examples = 0?"}
    none["no examples"]
    sel{"module:function<br/>selector?"}
    custom["call selector(rubric, calibration_pool,<br/>student_text, criterion)"]
    pool["criterion's pool entries<br/>(falls back to holistic entries)"]
    strat{"strategy"}
    random["random<br/>seeded per essay"]
    lexical["lexical<br/>BM25"]
    semantic["semantic<br/>embedding cosine"]
    hybrid["hybrid<br/>dense × w + BM25 × (1 − w)"]
    bucket["group score points into levels,<br/>take n_ex_per_score best per level,<br/>cap at max_examples"]
    out(["examples rendered into the prompt"])

    start --> zs
    zs -- yes --> none
    zs -- no --> sel
    sel -- yes --> custom --> out
    sel -- no --> pool --> strat
    strat --> random --> bucket
    strat --> lexical --> bucket
    strat --> semantic --> bucket
    strat --> hybrid --> bucket
    bucket --> out
```

Randomness is seeded from the rubric, the essay text, and `seed`, so a rerun picks the same
examples. Embeddings and BM25 indexes are cached under `<output_dir>/cache/retrieval/`, keyed by
a hash of the pool's IDs, scores, criteria, and text, so a changed pool can't reuse stale
vectors. `semantic` and `hybrid` need the `dense` extra.

---

[← Configuration resolution](02-configuration-resolution.md) · [Overview](README.md) · [Grading architectures →](04-grading-architectures.md)
