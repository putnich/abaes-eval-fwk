# Replication package

Per-essay result CSVs backing the paper's RQ1–RQ4 experiments, for recomputing metrics and
significance tests with the public `abaes-eval-fwk` framework. No aggregated tables or
precomputed metrics are included here — every number in the paper is recomputed by running
the commands below against these CSVs.

## Layout

Every file follows `RQ<n>_<architecture>_<...>_<model>[_{r,nr}].csv` — architecture and model
are always in the name, not just the directory, so a file is self-describing even copied out
of context. `_r` / `_nr` mark a model run with reasoning on / off.

```
results/
  RQ1/                        retrieval-strategy comparison, holistic scoring, Llama 3.3 only
    holistic/RQ1_holistic_<retrieval>_llama.csv   retrieval in {zero_shot, random, lexical, semantic, hybrid}
  RQ2/                        architecture comparison (holistic vs. criterion), hybrid retrieval, reasoning off
    holistic/RQ2_holistic_<model>.csv
    criterion/RQ2_criterion_<model>.csv
  RQ3/                        reasoning on/off ablation, hybrid retrieval -- the reasoning-on counterpart to RQ2
    holistic/RQ3_holistic_<model>_r.csv
    criterion/RQ3_criterion_<model>_r.csv
  RQ4/                        final-agent deliberation (no Baseline H/C files -- see below)
    holistic/RQ4_holistic_disagreement_based_adjudication_gemma_r.csv   DA-H
    holistic/RQ4_holistic_review_based_adjudication_gemma_r.csv         RA-H
    criterion/RQ4_criterion_disagreement_based_adjudication_gemma_r.csv DA-C
    criterion/RQ4_criterion_review_based_adjudication_gemma_r.csv       RA-C
    review/holistic/RQ4_holistic_review_gemma_r_<source_model>.csv      RA reviewer verdicts (outcome only)
    review/criterion/RQ4_criterion_review_gemma_r_<source_model>.csv
    disagreement/holistic/RQ4_holistic_disagreement_pool_gemma_r.csv    DA pool scores + disagreement outcome
    disagreement/criterion/RQ4_criterion_disagreement_pool_gemma_r.csv  (no graded scores -- excluded from metrics recomputation below)
```

`RQ2`'s six models (`llama`, `mixtral`, `phi`, `gemma_nr`, `qwen_nr`, `deepseek_nr`) are
reasoning disabled or not applicable; `RQ3`'s three models (`deepseek_r`, `gemma_r`, `qwen_r`)
are the same families with reasoning enabled. `RQ4`'s `<source_model>` names a member of the
pool Gemma 4 (reasoning on) adjudicates or reviews over -- each model's best RQ2/RQ3
configuration: `llama`, `deepseek_r`, `qwen_r` in the holistic setting, and `llama`,
`deepseek_nr`, `qwen_r` in the criterion setting.

## Setup

1. Obtain your own copies of ASAP AES, ASAP 2.0, and ELLIPSE (see `docs/datasets.md`);
   dataset access is subject to the original providers' terms and is not bundled here or in
   the framework. Lay them out as:

   ```
   <data_root>/datasets/ASAP_AES/training_set_rel3.tsv
   <data_root>/datasets/ASAP_2.0/ASAP2_train_sourcetexts.csv
   <data_root>/datasets/ELLIPSE/ELLIPSE_Final_github_train.csv
   <data_root>/datasets/ELLIPSE/ellipsis_raw_rater_scores_anon_all_essay.csv
   ```

2. From an `abaes-eval-fwk` checkout, install and generate the study project:

   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev]'
   abaes-eval-fwk init study --template study
   ```

3. Point `study/config/paths.yaml` at your `<data_root>` from step 1 (the only edit required —
   `study/config/datasets.yaml` already matches that layout):

   ```yaml
   data_root: /absolute/path/to/your/data_root
   output_dir: runs
   ```

4. Validate:

   ```bash
   abaes-eval-fwk validate study/benchmark.yaml --check-data
   ```

   This should print a validated submission count for each of the 6 datasets and
   `Valid configuration: 23 experiments, 6 datasets`, with no further edits needed.

`study/benchmark.yaml` is the `<config>` argument used in every command below.

## Recomputing metrics (RQ1–RQ3)

```bash
abaes-eval-fwk metrics study/benchmark.yaml results/study/RQ1/holistic/RQ1_holistic_hybrid_llama.csv --output analysis/rq1-hybrid
```

Reports NMAE, NRMSE, Pearson, Spearman, and QWK per dataset plus the unweighted average —
NMAE is what the paper reports; the rest are additional metrics over the same scores.

To recompute every file at once, nest `--output` by relative path, not bare filename — DA/RA's
holistic and criterion CSVs share the same basename by design, so nesting by filename alone
collides between the two:

```bash
find results/study \( -path '*/review/*' -o -path '*/disagreement/*' \) -prune -o -name '*.csv' -print | while read -r csv; do
  outdir="analysis/$(dirname "$csv" | sed 's#results/study/##')"
  abaes-eval-fwk metrics study/benchmark.yaml "$csv" --output "$outdir"
done
```

## Recomputing significance tests

Paired two-sided Wilcoxon tests over absolute scoring error, matched by essay ID:

```bash
abaes-eval-fwk compare study/benchmark.yaml \
  results/study/RQ1/holistic/RQ1_holistic_zero_shot_llama.csv results/study/RQ1/holistic/RQ1_holistic_hybrid_llama.csv \
  --output analysis/rq1_zero_shot_vs_hybrid.csv
```

Use the same pattern for any RQ2 holistic-vs-criterion pair (same model), RQ3
reasoning-off-vs-on pair (same architecture), or RQ4 baseline-vs-DA / baseline-vs-RA pair.

## Recomputing the RQ4 pool-average baselines

Each final-agent run's initial pool is the independent scores from three models over the same
essays (see the paper, Section on Final Agent Deliberation). Rebuild Baseline H / Baseline C
directly from the RQ2/RQ3 CSVs above:

```bash
# Baseline H: Llama 3.3, DeepSeek-R1-32B (reasoning on), Qwen 3.8 (reasoning on)
abaes-eval-fwk baseline study/benchmark.yaml \
  results/study/RQ2/holistic/RQ2_holistic_llama.csv \
  results/study/RQ3/holistic/RQ3_holistic_deepseek_r.csv \
  results/study/RQ3/holistic/RQ3_holistic_qwen_r.csv \
  --output analysis/pool_average_holistic.csv

# Baseline C: Llama 3.3, DeepSeek-R1-32B (reasoning off), Qwen 3.8 (reasoning on)
abaes-eval-fwk baseline study/benchmark.yaml \
  results/study/RQ2/criterion/RQ2_criterion_llama.csv \
  results/study/RQ2/criterion/RQ2_criterion_deepseek_nr.csv \
  results/study/RQ3/criterion/RQ3_criterion_qwen_r.csv \
  --output analysis/pool_average_criterion.csv
```

The DA-H/DA-C/RA-H/RA-C result CSVs under `RQ4/` are the final agent's (Gemma 4, reasoning
enabled) output over that same pool; they can be evaluated with `metrics` and compared against
baseline with `compare` as above, but regenerating them from scratch via `abaes-eval-fwk run`
requires access to that model.
