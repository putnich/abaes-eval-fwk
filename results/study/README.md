# Study results

Per-essay result CSVs from the RQ1–RQ4 study. Every reported metric, significance test, and RQ4
procedure statistic can be recomputed from these files with `abaes-eval-fwk`. No aggregated
tables or precomputed metrics are included.

## Layout

File names follow `RQ<n>_<architecture>_<...>_<model>[_{r,nr}].csv`, so a file is self-describing
outside its folder. `_r` / `_nr` mark a model run with reasoning on / off.

```
results/study/
  RQ1/holistic/RQ1_holistic_<retrieval>_llama.csv       retrieval: zero_shot, random, lexical, semantic, hybrid
  RQ2/{holistic,criterion}/RQ2_<arch>_<model>.csv       architecture comparison; hybrid retrieval, reasoning off
  RQ3/{holistic,criterion}/RQ3_<arch>_<model>_r.csv     reasoning-on counterparts of RQ2
  RQ4/{holistic,criterion}/
    RQ4_<arch>_disagreement_based_adjudication_gemma_r.csv   DA
    RQ4_<arch>_review_based_adjudication_gemma_r.csv         RA
  RQ4/review/{holistic,criterion}/RQ4_<arch>_review_gemma_r_<source_model>.csv   RA review verdicts
  RQ4/disagreement/{holistic,criterion}/RQ4_<arch>_disagreement_pool_gemma_r.csv DA pool scores and outcome
  samples/rq4_review_demo_<arch>.csv                    sample reviews with full justifications and reasoning
```

- **RQ2** models: `llama`, `mixtral`, `phi`, `gemma_nr`, `qwen_nr`, `deepseek_nr` (reasoning off
  or not applicable).
- **RQ3** models: `deepseek_r`, `gemma_r`, `qwen_r` (the same families with reasoning on).
- **RQ4**: Gemma 4 (reasoning on) adjudicates or reviews a pool of three source models, each in
  its best RQ2/RQ3 configuration: `llama`, `deepseek_r`, `qwen_r` (holistic) and `llama`,
  `deepseek_nr`, `qwen_r` (criterion).

`review/`, `disagreement/`, and `samples/` hold no graded scores; skip them when recomputing
metrics.

## Setup

1. Download ASAP AES, ASAP 2.0, and ELLIPSE as described in
   [docs/datasets.md](../../docs/datasets.md#downloading-the-benchmark-corpora).
2. Install the framework and create the study project:

   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev]'
   abaes-eval-fwk init study --template study
   ```

3. Point `study/config/paths.yaml` at the directory that contains `datasets/`:

   ```yaml
   data_root: /absolute/path/to/data_root
   output_dir: runs
   ```

4. Validate:

   ```bash
   abaes-eval-fwk validate study/benchmark.yaml --check-data
   ```

   Expect a submission count for each of the 6 datasets and
   `Valid configuration: 23 experiments, 6 datasets`.

## Metrics (RQ1–RQ3, RQ4 DA/RA)

```bash
abaes-eval-fwk metrics study/benchmark.yaml \
  results/study/RQ1/holistic/RQ1_holistic_hybrid_llama.csv --output analysis/rq1-hybrid
```

Reports NMAE (the primary metric), NRMSE, Pearson, Spearman, and QWK per dataset, plus the
dataset-equal average. Each input gets its own folder under `--output`, named after the CSV, so
every file can be recomputed into one directory:

```bash
find results/study \( -path '*/review/*' -o -path '*/disagreement/*' -o -path '*/samples/*' \) -prune \
  -o -name '*.csv' -print | while read -r csv; do
  abaes-eval-fwk metrics study/benchmark.yaml "$csv" --output analysis/metrics
done
```

## Significance tests

Paired two-sided Wilcoxon tests on normalized absolute scoring error, matched by essay ID:

```bash
abaes-eval-fwk compare study/benchmark.yaml \
  results/study/RQ1/holistic/RQ1_holistic_zero_shot_llama.csv \
  results/study/RQ1/holistic/RQ1_holistic_hybrid_llama.csv \
  --output analysis/rq1_zero_shot_vs_hybrid.csv
```

Use the same pattern for RQ2 holistic vs. criterion (same model), RQ3 reasoning off vs. on (same
architecture), and RQ4 baseline vs. DA or RA.

## RQ4 pool-average baselines

Baseline H and Baseline C average the three pool members' independent scores, rebuilt from the
RQ2/RQ3 CSVs:

```bash
abaes-eval-fwk baseline study/benchmark.yaml \
  results/study/RQ2/holistic/RQ2_holistic_llama.csv \
  results/study/RQ3/holistic/RQ3_holistic_deepseek_r.csv \
  results/study/RQ3/holistic/RQ3_holistic_qwen_r.csv \
  --output analysis/pool_average_holistic.csv

abaes-eval-fwk baseline study/benchmark.yaml \
  results/study/RQ2/criterion/RQ2_criterion_llama.csv \
  results/study/RQ2/criterion/RQ2_criterion_deepseek_nr.csv \
  results/study/RQ3/criterion/RQ3_criterion_qwen_r.csv \
  --output analysis/pool_average_criterion.csv
```

The DA/RA CSVs are Gemma 4's output over the same pools. Regenerating them with
`abaes-eval-fwk run` needs access to that model.

## RQ4 procedure outcomes and review tokens

`adjudication` takes the same pool sources as `baseline` plus the pool's review CSVs. It recomputes
each DA outcome with the run-time rule (normalized mean pairwise disagreement above 0.2) and each
RA outcome from the reviews (positive: all assessments admissible; mixed: some; negative: none):

```bash
abaes-eval-fwk adjudication study/benchmark.yaml \
  results/study/RQ2/holistic/RQ2_holistic_llama.csv \
  results/study/RQ3/holistic/RQ3_holistic_deepseek_r.csv \
  results/study/RQ3/holistic/RQ3_holistic_qwen_r.csv \
  --reviews results/study/RQ4/review/holistic/*.csv \
  --output analysis/adjudication_holistic

abaes-eval-fwk adjudication study/benchmark.yaml \
  results/study/RQ2/criterion/RQ2_criterion_llama.csv \
  results/study/RQ2/criterion/RQ2_criterion_deepseek_nr.csv \
  results/study/RQ3/criterion/RQ3_criterion_qwen_r.csv \
  --reviews results/study/RQ4/review/criterion/*.csv \
  --output analysis/adjudication_criterion
```

Each output directory contains:

- `outcomes.csv`: count and share of each DA and RA outcome per dataset and over `All items`
  (1,200 holistic pools, 5,400 criterion-level pools).
- `outcome_items.csv`: each pool's outcome, DA's normalized disagreement, and RA's admissible
  count.
- `review_tokens.csv`: average review completion tokens per essay for each reviewed source
  model, summed over an essay's criteria and averaged with datasets weighted equally.

Expected `All items` and `Average` values (the reported values):

| | Holistic | Criterion |
| --- | --- | --- |
| DA agreement | 84.7% | 86.0% |
| RA positive (all admissible) | 36.7% | 35.6% |
| RA mixed | 56.0% | 62.2% |
| RA negative | 7.3% | 2.2% |
| Review tokens, DeepSeek-R1-32B | 1,330 | 5,210 |
| Review tokens, Llama 3.3 | 1,306 | 5,446 |
| Review tokens, Qwen 3.8 | 1,060 | 4,147 |

`tests/test_replication_package.py` pins these values. It also checks that resolving each pool
from its counted outcome, with Gemma 4's RQ3 assessment as the independent one, rebuilds every
final score in the DA and RA CSVs.
