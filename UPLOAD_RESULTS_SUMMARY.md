# Upload Results Summary

## Scope

This folder contains the code, notebooks, and lightweight result artifacts intended for GitHub upload.

Original remote repository files were not modified. This is a separate upload-ready copy.

## Code Files To Upload

- `10_civil_comments_identity_bias.ipynb`
- `11_reviews_yelp_amazon_sentiment.ipynb`
- `12_goemotions_emotion_bias.ipynb`
- `13_mnli_genre_nli_bias.ipynb`
- `14_long_context_civil_longformer.ipynb`
- `train.py`
- `config.py`
- `requirements-colab.txt`
- `COLAB_RUNBOOK.md`
- `EXPERIMENT_PLAN.md`
- `README_APPEND_ONLY.md`
- `aix_bias_suite/__init__.py`
- `aix_bias_suite/runtime.py`
- `tools/repair_notebooks.py`
- `tools/validate_notebooks.py`

## Result Files To Upload

Result root:

```text
results/civil_39run_9h_a100_resume/
```

Included result files:

- `report.md`
- 13 CSV files in `tables/`
- 7 PNG files in `figures/`

Excluded result files:

- `checkpoints/`
- `runs/`
- `training_logs/`
- `civil_39run_9h_a100_resume.zip`
- `__pycache__/`

Reason: these excluded files are either large model/runtime artifacts or too verbose for a clean GitHub upload. The included CSV, PNG, and report files preserve the reviewable experiment result.

## Completed Result Snapshot

- Experiment result folder: `civil_39run_9h_a100_resume`
- Completed runs in `final_ranking.csv`: 39
- Device used for the completed result: NVIDIA A100-SXM4-80GB
- Run profile: `h100_long`
- Task: CivilComments toxicity and identity-bias evaluation
- Models evaluated:
  - `microsoft/deberta-v3-large`
  - `FacebookAI/roberta-large`
  - `FacebookAI/xlm-roberta-large`

## Best Composite Run

```text
run_id: microsoft__deberta-v3-large__identity_mentioned__lora_r8
base_model: microsoft/deberta-v3-large
domain: identity_mentioned
method: lora
lora_rank: 8
avg_test_f1_macro: 0.7982494474145287
avg_test_accuracy: 0.9199999999999999
bias_score: 0.5583333333333333
composite_score: 0.8067688956768602
training_time_seconds: 225.25368928909302
peak_gpu_memory_mb: 5121.873046875
```

## File Rename Mapping

The notebook names were shortened for GitHub readability. `config.py` was updated so `train.py` still resolves the correct notebooks by task.

```text
01_h100_heavy_civil_comments_identity_bias.ipynb -> 10_civil_comments_identity_bias.ipynb
02_h100_heavy_reviews_yelp_amazon_sentiment.ipynb -> 11_reviews_yelp_amazon_sentiment.ipynb
03_h100_heavy_goemotions_emotion_bias.ipynb -> 12_goemotions_emotion_bias.ipynb
04_h100_heavy_mnli_genre_nli_bias.ipynb -> 13_mnli_genre_nli_bias.ipynb
05_h100_heavy_long_context_civil_longformer.ipynb -> 14_long_context_civil_longformer.ipynb
```

## Suggested Commit Message

```text
Add H100 heavy bias notebooks and CivilComments A100 results
```

Alternative:

```text
Add Colab heavy bias suite with CivilComments result summary
```
