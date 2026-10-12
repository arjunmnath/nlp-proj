# Run order

These are separate experimental pipelines. Run them independently because their preprocessing and validation sources differ in the original notebooks.

```bash
PYTHONPATH=src python scripts/run_imbalance.py --train-path DATA/mal_full_offensive_train.csv --dev-path DATA/mal_full_offensive_dev.xlsx --with-augmentation
PYTHONPATH=src python scripts/run_llm_augmentation.py --train-path DATA/mal_full_offensive_train.csv --dev-path DATA/mal_full_offensive_dev.csv --num-samples 5
PYTHONPATH=src python scripts/run_combined_olid.py --mal-train DATA/mal_full_offensive_train.csv --mal-dev DATA/mal_full_offensive_dev.csv --olid-train DATA/olid-training-v1.0.tsv
```
