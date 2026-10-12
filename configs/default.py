from pathlib import Path

# Paths are intentionally empty: pass dataset paths through the CLI.
MAL_TRAIN = Path("data/FIRE-2025/Malayalam/mal_full_offensive_train.csv")
MAL_DEV_XLSX = Path("data/FIRE-2025/Malayalam/mal_full_offensive_dev.xlsx")
MAL_DEV_CSV = Path("data/FIRE-2025/Malayalam/mal_full_offensive_dev.csv")
OLID_TRAIN = Path("data/OLID/olid-training-v1.0.tsv")

SEED = 42
MODEL_PATH = "setu4993/LaBSE"

IMBALANCE = {
    "max_len": 128,
    "batch_size": 32,
    "eval_batch_size": 128,
    "epochs": 4,
    "head_lr": 5e-4,
    "enc_lr": 2e-5,
    "llrd": 0.9,
    "weight_decay": 0.01,
    "unfreeze_schedule": {0: 1, 1: 2, 2: 4, 3: 6},
    "gamma": 2.0,
    "beta": 0.999,
    "aug_ratio": 0.25,
    "aug_max_per_sample": 6,
    "aug_max_total": 20000,
    "nllb_model": "facebook/nllb-200-distilled-600M",
    "posthoc_tau": 0.5,
}

LLM_AUG = {
    "generator_model": "sarvamai/sarvam-1-v0.5",
    "classifier_encoder": "sentence-transformers/LaBSE",
    "few_shot_size": 50,
    "few_shot_random_state": 42,
    "max_length": 128,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "generation_temperature": 0.8,
    "generation_top_p": 0.9,
    "classifier_batch_size": 32,
    "classifier_lr": 1e-3,
    "classifier_weight_decay": 1e-4,
    "classifier_epochs": 15,
}

COMBINED = {
    "max_len": 256,
    "batch_size": 32,
    "eval_batch_size": 128,
    "learning_rate": 3e-5,
    "epochs": 4,
}
