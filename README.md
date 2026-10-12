# Malayalam Offensive Language Classification

A reproducible project aggregating three experimental approaches from the supplied notebooks for the same 5-class offensive-language classification problem using **FIRE-2025 Malayalam** and **OLID**:

1. **Imbalance-aware LaBSE fine-tuning** — cross-entropy / focal loss, class-balanced focal loss, weighted sampling, minority augmentation, and post-hoc prior adjustment.
2. **LLM-generated minority augmentation + LaBSE embeddings** — fine-tune Sarvam-1 with LoRA on 50 inverse-frequency examples, generate labelled synthetic Malayalam samples, append them to training data, then train an MLP over LaBSE sentence embeddings.
3. **Malayalam + OLID joint training** — concatenate Malayalam and OLID training sets, use LaBSE with mean/max pooling and class-weighted cross-entropy, and evaluate on the Malayalam development set.

The original notebooks are retained under `original_notebooks/` for provenance.

## Important reproducibility constraint

The seed values from the notebooks are preserved:

- Main imbalance experiments: `42`
- LLM augmentation few-shot sampling: `random_state=42`
- Combined Malayalam+OLID: `RANDOM_SEED = 42`

Do **not** change these values if you need to reproduce the notebook results. The LLM generation stage is inherently sampling-based; its generation parameters are kept as in the source notebook.

## Dataset

The source experiments use:

```text
FIRE-2025/Malayalam/mal_full_offensive_train.csv
FIRE-2025/Malayalam/mal_full_offensive_dev.xlsx   # imbalance notebook
FIRE-2025/Malayalam/mal_full_offensive_dev.csv    # LLM/combined notebooks
OLID/olid-training-v1.0.tsv
```

Set the paths in `configs/default.py`, or pass them on the command line. The project does not redistribute the datasets.

The common class mapping is:

| ID | Class |
|---:|---|
| 0 | `Not_offensive` |
| 1 | `Offensive_Untargetede` |
| 2 | `Offensive_Targeted_Insult_Individual` |
| 3 | `Offensive_Targeted_Insult_Group` |
| 4 | `Offensive_Targeted_Insult_Other` |

## Environment

Python 3.10+ with a CUDA-capable PyTorch installation is recommended. The original notebooks were run in Kaggle/Colab-style GPU environments.

```bash
pip install -r requirements.txt
```

For the LLM augmentation approach, also ensure enough GPU memory for `sarvamai/sarvam-1-v0.5`; the original notebook used CUDA and FP16. NLLB augmentation in the imbalance approach requires additional GPU memory.

## Project layout

```text
malayalam_offensive_project/
├── README.md
├── requirements.txt
├── configs/
│   └── default.py
├── src/
│   └── offensive_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       ├── labse.py
│       ├── imbalance.py
│       ├── augmentation.py
│       ├── llm_augmentation.py
│       └── combined.py
├── scripts/
│   ├── run_imbalance.py
│   ├── run_llm_augmentation.py
│   └── run_combined_olid.py
└── original_notebooks/
    ├── imbalance_experiments.ipynb
    ├── llm_spell_augmentation.ipynb
    └── malayalam_olid_combined.ipynb
```

## Approach 1 — imbalance experiments

This is the modular form of the `notebook2a20ed4195.ipynb` experiments. It preserves the original configuration:

- LaBSE: `setu4993/LaBSE`
- max length: 128
- batch size: 32
- evaluation batch size: 128
- epochs: 4
- head LR: `5e-4`
- encoder LR: `2e-5`
- LLRD: `0.9`
- weight decay: `0.01`
- focal gamma: `2.0`
- CB beta: `0.999`
- unfreezing schedule: `{0: 1, 1: 2, 2: 4, 3: 6}`
- seed: `42`

Experiments:

```text
R0  CE + random sampling
R1  Focal + WeightedRandomSampler
R2  CB-Focal + random sampling
R3  CB-Focal + WeightedRandomSampler
R4  CB-Focal + WeightedRandomSampler + minority augmentation
R4b CB-Focal + minority augmentation + random sampling
R5  CB-Focal + sqrt-inverse WeightedRandomSampler
```

Run:

```bash
PYTHONPATH=src python scripts/run_imbalance.py \
  --train-path /path/to/mal_full_offensive_train.csv \
  --dev-path /path/to/mal_full_offensive_dev.xlsx
```

Outputs are written to `outputs/imbalance/`.

## Approach 2 — LLM-generated minority augmentation

This corresponds to `spell_research.ipynb`.

The pipeline is:

```text
Malayalam train data
        ↓
inverse-frequency sample of 50 examples (random_state=42)
        ↓
Sarvam-1 LoRA fine-tuning
        ↓
LLM-generated labelled samples
        ↓
original + synthetic training set
        ↓
LaBSE sentence embeddings
        ↓
256 → 64 → 5 MLP classifier
        ↓
Malayalam dev evaluation
```

Run:

```bash
PYTHONPATH=src python scripts/run_llm_augmentation.py \
  --train-path /path/to/mal_full_offensive_train.csv \
  --dev-path /path/to/mal_full_offensive_dev.csv \
  --num-samples 5
```

The LoRA parameters remain those in the notebook:

```text
r=8
lora_alpha=16
lora_dropout=0.05
target_modules=[q_proj, v_proj]
learning_rate=2e-4
epochs=1
per_device_train_batch_size=1
gradient_accumulation_steps=4
max_length=128
```

The classifier remains:

```text
LaBSE embedding
    → Linear(768, 256)
    → ReLU
    → Dropout(0.3)
    → Linear(256, 64)
    → ReLU
    → Dropout(0.3)
    → Linear(64, 5)
```

## Approach 3 — Malayalam + OLID

This corresponds to `transformer-combined-2.ipynb`.

The training dataset is:

```text
Malayalam training set + OLID training set
```

while validation remains the Malayalam development set.

The original setup is preserved:

- LaBSE: `setu4993/LaBSE`
- max length: 256
- train batch size: 32
- validation batch size: 128
- LR: `3e-5`
- epochs: 4
- seed: 42
- weighted CE: `1 + log(N / class_count)`
- LaBSE mean pooling + max pooling
- linear classifier over the 1536-dimensional pooled representation

Run:

```bash
PYTHONPATH=src python scripts/run_combined_olid.py \
  --mal-train /path/to/mal_full_offensive_train.csv \
  --mal-dev /path/to/mal_full_offensive_dev.csv \
  --olid-train /path/to/olid-training-v1.0.tsv
```

## Reproducing results

1. Install the dependencies.
2. Download the same dataset versions used by the notebooks.
3. Use the same pretrained model identifiers.
4. Keep all seed values at `42`.
5. Run the corresponding script without changing hyperparameters.
6. Compare the generated reports under `outputs/` with the notebook outputs.

For exact notebook provenance, compare the modular implementation against the copies in `original_notebooks/`.

## Notes on dataset differences

The imbalance notebook uses the official Malayalam Excel development file and performs additional cleaning, deduplication, and train/validation overlap removal. The other two notebooks use a CSV development file and their own preprocessing. These are deliberately kept as separate pipelines rather than silently forcing one preprocessing scheme onto all three approaches.

The imbalance notebook also contains a five-class `OTH` category that has zero training and validation support in the observed Malayalam split. Metrics therefore report both the present-class macro-F1 and the fixed five-class macro-F1 where applicable.
