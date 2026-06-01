# DynamicFold

A deep learning pipeline that integrates RNA sequence and transcriptomic data to predict dynamic RNA secondary structures in living cells. DynamicFold trains on paired RNA-seq and icSHAPE data to predict in-vivo icSHAPE reactivity profiles, then feeds those predictions into thermodynamic folding tools to reconstruct cell-state-specific secondary structures.

## Overview

RNA secondary structure is not static — it varies across cell types, developmental stages, and conditions. DynamicFold captures this dynamicity by:

1. **Building a multimodal dataset** from paired RNA-seq and icSHAPE experiments across human neurogenesis (D0/D7/D8/D14) and zebrafish embryogenesis (2h/4h/6h, wild-type and Elavl1a mutant).
2. **Training sequence models** that take RNA sequence features alongside transcriptomic signals (read depth, cleavage rate, mismatch rate) and predict per-nucleotide icSHAPE reactivity, achieving a best test MAE of ~0.14.
3. **Predicting dynamic structures** by running the predicted reactivity through ViennaRNA/RNAfold (Deigan method), benchmarked against RibonanzaNet and no-SHAPE baselines.

## Repository Structure

```
.
├── data/
│   ├── neural/process/         # Human neurogenesis dataset pipeline
│   │   ├── autopilot.sh        # End-to-end build script
│   │   ├── fetch.py            # SRA download
│   │   ├── rnaseq.py           # RNA-seq alignment & metrics
│   │   ├── icshape.py          # icSHAPE alignment & RT-stop counting
│   │   ├── format.py           # Assemble per-transcript dataset entries
│   │   ├── depths.py           # Compute per-SRR total read depths
│   │   ├── references.py       # Populate reference annotation table
│   │   ├── replicates.py       # PCA replicate QC
│   │   ├── sam.py              # SAM/BAM parser (metrics & RT-stops)
│   │   ├── utils.py            # Label, Database, Executer utilities
│   │   └── genome/             # Transcriptome construction scripts
│   ├── zebrafish/process/      # Zebrafish embryogenesis dataset pipeline
│   │   └── ...                 # Mirrors neural/ structure
│   └── assembly/process/       # Cross-dataset assembly
│       ├── autopilot.sh        # Merge, filter, sample, analyse
│       ├── assembly.py         # Merge source DBs into assembly.db
│       ├── query.py            # SQL → CSV export
│       ├── update.py           # Merge external columns into CSV
│       └── insights/           # Histogram, completeness, MI analysis
│
├── models/
│   ├── hybrid/                 # Custom U-Net + attention architectures
│   │   ├── H08.py              # GRU baseline
│   │   ├── H15.py              # Conv-GRU-Attention U-Net
│   │   ├── H16.py              # Conv blocks + cross-attention memory
│   │   ├── H20.py              # Conv U-Net + transformer bottleneck
│   │   ├── H21.py              # Conv U-Net + 2-D contact-map bottleneck
│   │   ├── main.py             # Training/evaluation entry point
│   │   ├── utils.py            # Dataset, Checkpoint, Trainer
│   │   ├── modules.py          # Reusable nn.Module building blocks
│   │   └── {H*/}/              # Per-run configs.json & outputs
│   ├── basewise/               # Sklearn baselines (LR, RF, XGB, MLP, SVR)
│   │   └── {LR,RF,XGB,MLP,SVR}/
│   └── RibonanzaNet/           # RibonanzaNet inference wrapper
│       ├── infer.py / infer.ipynb
│       └── configs/
│
│
└── results/
    ├── dataset/                # Dataset characterisation plots & scripts
    │   ├── features/           # Length, depth, reactivity distributions
    │   ├── sequences/          # Base composition, motif coverage, biotype
    │   └── pairs/              # Pairwise feature KDE heatmaps
    ├── performance/            # Model evaluation
    │   ├── residual/           # Basewise error analysis
    │   ├── performance/        # MAE vs sequence properties
    │   └── feature/            # Saliency maps, hidden-state PCA, ablation
    └── dynamic/                # Dynamic structure benchmarking
        ├── predict_structure.py
        ├── curves.py           # ROC / PR curves
        ├── tables.py           # F1 / precision / recall table
        ├── dynamicity_squares.py
        ├── transcript_mae.py
        └── selection.py        # Case-study sequence selection
```

## Key Features

- **Multimodal input** — one-hot sequence (A/C/G/U) + log-scaled RNA-seq read depth + cleavage rate + mismatch rate, optionally augmented with RibonanzaNet in-vitro predictions.
- **Length-aware batching** — sequences vary from 64 to 4096 nt; a `LengthAwareSampler` groups similar lengths with jitter to minimise padding waste.
- **Reproducible checkpointing** — each training run stores an MD5 checksum-based serial, parent lineage, loss history, and model/optimizer state in a single `.pt` file.
- **SQLite-backed dataset** — raw alignment data and final dataset entries are stored in SQLite for fast keyed access; the assembly DB is write-protected after build.
- **End-to-end structure pipeline** — predicted reactivities are fed to ViennaRNA via the Deigan method; outputs benchmarked with ROC/PR curves and F1 against experimental dot-bracket structures.

## Installation

### Requirements

Python ≥ 3.11, CUDA-capable GPU recommended.

```bash
# Core dependencies
pip install torch numpy pandas scikit-learn tqdm orjson

# Bioinformatics
pip install pysam biopython

# RNA folding
pip install ViennaRNA          # or install via conda-forge

# Extra
pip install xgboost einops adjustText statsmodels seaborn
```

External tools (must be on `PATH`): `bowtie2`, `bowtie2-build`, `trim_galore`, `cutadapt`, `clumpify.sh` (BBTools), `prefetch` / `fasterq-dump` (SRA Toolkit), `pigz`, `gffread`, `vmtouch`.

## Usage

### 1. Build the dataset

```bash
# Human neurogenesis
cd data/neural/process
bash autopilot.sh          # aligns reads, builds neural.db

# Zebrafish embryogenesis
cd data/zebrafish/process
bash autopilot.sh          # builds zebrafish.db

# Assemble & filter
cd data/assembly/process
bash autopilot.sh          # merges both DBs → assembly.db + assembly.csv
```

### 2. Train a model

```bash
cd models/hybrid
python main.py -c H21/1/configs.json          # train from scratch
python main.py -c H21/1/configs.json -d cuda  # specify device
```

Config files follow this schema:

```json
{
  "Mode": "New",          // New | Resume | Transfer | Evaluate
  "Module": "H21",        // Python module name
  "Model":  "H21",        // Class name inside that module
  "Optimizer": "Adam",
  "DatasetArgs": { ... },
  "ModelArgs":  { ... },
  "OptimizerArgs": { ... },
  "AutopilotArgs": {
    "loss_fn": "L1Loss",
    "max_epochs": 64,
    "output_dir": "H21/1"
  }
}
```

Resume training or run evaluation by changing `"Mode"` and providing `"CheckpointPT"`.

### 3. Run RibonanzaNet inference

```bash
cd models/RibonanzaNet
python infer.py configs/configs.json
```

Predictions are written back into the assembly SQLite database.

### 4. Predict secondary structures & benchmark

```bash
cd results/dynamic
python predict_structure.py <data.csv>   # adds dot-bracket columns
python curves.py data.csv 200 results/curves
python tables.py data.csv results/table.csv
```

### 5. Evaluate model performance

```bash
cd results/performance/feature
python feature_saliency.py configs.json saliency.png
python hidden_states.py configs.json 5000 4 4 pca_states
python saliency_map.py configs.json ENST00000439929 maps/
```

## Data Schema

Each row in the assembled dataset (`assembly.csv` / `assembly.db`) contains:

| Column | Type | Description |
|---|---|---|
| `SeqID` | str | `{Sample}\|{RefName}` |
| `A/C/G/U` | JSON list | One-hot channels |
| `RD` | JSON list | Log₁₀ RNA-seq read depth per base |
| `ER` | JSON list | −Log₁₀ cleavage rate per base |
| `MR` | JSON list | −Log₁₀ mismatch rate per base |
| `RT` | JSON list | Winsorized+scaled icSHAPE reactivity (label) |
| `IC` | JSON list | Missing-data indicator bitmask per base |
| `Sequence` | str | RNA sequence (U alphabet) |
| `FullLength` | int | Total transcript length |
| `ValidLength` | int | Length of covered region |
| `MeanDepth` | float | Mean RNA-seq CPM over valid region |
| `MeanDensity` | float | Mean icSHAPE DMSO base density |
| `Gap` | int | Bases with missing data inside valid region |

## Results Summary

| Model | Test MAE |
|---|---|
| Linear Regression | ~0.21 |
| Random Forest | ~0.19 |
| RibonanzaNet (in-vitro) | ~0.18 |
| H08 (GRU) | ~0.16 |
| H20 (Conv U-Net) | ~0.15 |
| **H21 (Conv U-Net + 2D map)** | **~0.14** |

The best model improves secondary structure prediction F1 and AUC over both the no-SHAPE baseline and RibonanzaNet, and captures cell-state-dependent structural variability that static models cannot.

## License

Research use. See individual third-party components (RibonanzaNet, ViennaRNA) for their respective licenses.
