# Run order

These are separate experimental pipelines. Run them independently because their preprocessing and validation sources differ in the original notebooks.

```bash
PYTHONPATH=src python scripts/run_imbalance.py --train-path data/Malayalam/mal_full_offensive_train.csv --dev-path data/Malayalam/mal_full_offensive_dev.csv  --with-augmentation
PYTHONPATH=src python scripts/run_llm_augmentation.py --train-path data/Malayalam/mal_full_offensive_train.csv --dev-path data/Malayalam/mal_full_offensive_dev.csv --num-samples 5
PYTHONPATH=src python scripts/run_combined_olid.py --mal-train data/Malayalam/mal_full_offensive_train.csv --mal-dev data/Malayalam/mal_full_offensive_dev.csv --olid-train data/OLID/olid-training-v1.0.tsv
```
