# Final-agent procedures

How RQ4 experiments combine earlier assessments, and how they depend on the experiments that produced them.

A final agent works from the assessments that earlier experiments produced (the *candidates*).
Adjudicating procedures also need the final model's own *independent* assessment of each essay,
made earlier and without seeing the candidates. Every branch's score is decided
deterministically from those inputs. When the model is called, it's only to justify a score
already fixed, so it can't be talked into a candidate's number.

For `criterion_specialized` candidates, each criterion is adjudicated as its own item in one
shared conversation, followed by a synthesis turn. Holistic candidates are a single item.

## pool_average

The candidates' scores are averaged and rounded half-up to a score level. No model call. This
is the baseline the adjudicating procedures are compared against.

For `criterion_specialized` candidates, each candidate's criterion scores are first combined
into that candidate's overall score with the rubric's `criteria_aggregation`. The baseline is
then the half-up rounded mean of those overall scores. Criteria are never averaged across
candidates before aggregation, so a `sum` rubric with candidates totalling 0 and 2 scores 1.

## disagreement

```mermaid
flowchart TB
    s(["candidate scores for one item"])
    napd["NAPD = mean |a − b| over all pairs<br/>÷ (score_max − score_min) × 5"]
    t{"NAPD > disagreement_threshold × 5?<br/>(default 0.2 → 20% of the scale)"}
    agree["agreement:<br/>score = round_half_up(mean of candidates)<br/>model writes the justification"]
    dis["disagreement:<br/>candidates rejected,<br/>score = final model's independent assessment<br/>(no model call)"]
    s --> napd --> t
    t -- no --> agree
    t -- yes --> dis
```

NAPD (normalized average pairwise disagreement) puts every rubric on a common 0–5 reference
width, so one threshold works across scales of different sizes and any number of candidates.

## review_based

```mermaid
flowchart TB
    s(["candidate assessments for one item"])
    rev["final model reviews each candidate<br/>in its own conversation:<br/>rubric-consistent AND evidence-grounded?"]
    cache[("reviews CSV<br/>reused via reuse.reviews")]
    c{"how many admissible?"}
    pos["positive (all):<br/>score = mean of candidates"]
    mix["mixed (some):<br/>score = mean of admissible<br/>+ independent assessment"]
    neg["negative (none):<br/>score = independent assessment<br/>(no model call)"]
    just["model writes the justification<br/>for the fixed score"]
    s --> rev
    rev <--> cache
    rev --> c
    c --> pos --> just
    c --> mix --> just
    c --> neg
```

A review judges two things: whether the score is consistent with the rubric's level
guidelines, and whether the justification is grounded in evidence from the essay. A candidate is
admissible only if it passes both. Each review runs in a separate conversation so no verdict
anchors another. Reviews are written
to the run's reviews CSV. A `review` (`non_grading`) experiment produces them on their own, and a
later `review_based` experiment can replay them through `reuse.reviews` instead of asking again.

## Experiment dependencies

`reuse` entries can name earlier experiments as `@name`. Selecting an experiment schedules its
dependencies first, depth-first, and rejects cycles and unknown names. The playground's RQ4
experiments form this graph:

```mermaid
flowchart LR
    a["rq2-a-holistic<br/>model a"]
    b["rq2-b-holistic<br/>model b"]
    f["rq2-final-holistic<br/>final model"]
    pa["rq4-holistic-pool_average"]
    d["rq4-holistic-disagreement"]
    rb["rq4-holistic-review_based"]
    r["rq4-holistic-review<br/>non_grading"]
    rbc["rq4-holistic-review_based-cached"]

    a -- results --> pa
    b -- results --> pa
    a -- results --> d
    b -- results --> d
    f -- independent --> d
    a -- results --> rb
    b -- results --> rb
    f -- independent --> rb
    a -- results --> r
    r -- reviews --> rbc
    a -- results --> rbc
    b -- results --> rbc
    f -- independent --> rbc
```

`abaes-eval-fwk run benchmark.yaml --experiment rq4-holistic-disagreement --dry-run` prints this
order before anything runs.

---

[← Grading architectures](04-grading-architectures.md) · [Overview](README.md) · [Evaluation →](06-evaluation.md)
