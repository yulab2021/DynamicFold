## Metadata

- Journal: Nature Communications
- Type: Article
- Limit: 2000 words, 200 abstract words, 15 title, 4 figures, 70 references
- Title: **TODO**
- Model Name: DynamicFold

## Model Metadata

- Label: experimental icSHAPE data
- Features: sequence, read depth, end rate, mismatch rate, RN prediction
- Important Aspects: valid length, mean depth, mean density, biotype
- Optional Aspects: strip start, strip end, gap, mean end depth, mean mismatch rate, icSHAPE reactivity
- Unimportant Aspects: full length, strip length
- Optional Layer: RNA-Seq FPKM vector + MLP // potentially reflects cell type and cell state

## Figure 1: Dataset Characteristics and Model Training

- 1A: Diagram: Data processing/assembly workflow
- 1B: Diagram (selective supplementary): RNA-Seq, DMSO, NAIN3 coverage on a reference transcript, explaining valid range and metrics
- 1C: Multi-panel (selective supplementary)
  - Violin plot: Base type proportion w.r.t. dataset entry
  - Stacked bar plot: Motif coverage proportion w.r.t. motif length
  - Histogram: Sequence valid length distribution w.r.t. dataset entry
  - KDE plot: RNA-Seq mean depth distribution + icSHAPE mean background base density distribution w.r.t. dataset entry
  - KDE plot: icSHAPE reactivity + RN prediction distribution w.r.t. base
  - Pie chart: Entry biotype w.r.t. dataset entry
  - KDE plot: strip start + strip end + gap distribution w.r.t. dataset entry
  - Histogram: mean end rate distribution w.r.t. dataset entry
  - Histogram: mean mismatch rate distribution w.r.t. dataset entry
  - KDE heatmap: Label vs. features w.r.t. base
  - Violin plot: icSHAPE reactivity distribution w.r.t. base, grouped by base type
  - Violin plot & significance test: Features distribution w.r.t. base, grouped by rasterized label and features // rasterization: split by median, split by quartile
- 1D: Multi-panel line plot: Training + validation loss w.r.t. epoch for all deep models
- 1E: Violin plot: MAE comparison of basewise models (LR, SVR, RF, XGB) + deep models (H08~H13) + RN

## Figure 2: Model Performance Analysis

- 2A: Diagram: Architecture of best model
- 2B: Multi-panel: Residual analysis of best model (selective supplementary)
  - Scatter plot: True vs. predicted icSHAPE reactivity w.r.t. base
  - KDE plot: best model vs. RibonanzaNet Basewise error distribution w.r.t. base
  - KDE heatmap: Basewise error vs. read depth w.r.t. base
  - KDE heatmap: Basewise error vs. end rate & mismatch rate w.r.t. base
  - KDE heatmap: Basewise error vs. RN prediction & icSHAPE reactivity w.r.t. base
  - KDE heatmap: best model error vs. RibonanzaNetFold error w.r.t. base
  - Violin plot: Basewise error vs. base type w.r.t. base
- 2C: Multi-panel: Performance analysis of best model (selective supplementary)
  - Scatter plot: MAE vs. valid length w.r.t. dataset entry
  - KDE heatmap: MAE vs. RNA-Seq mean depth & icSHAPE mean density w.r.t. dataset entry
  - KDE plot: MAE distribution of best model + RibonanzaNet
  - Scatter plot: MAE vs. mean end depth + mean mismatch rate w.r.t. dataset entry
- 2D: Multi-panel: Feature importance analysis
  - Violin plot: Distribution of mean absolute normalized saliency w.r.t. dataset entry grouped by channel // mean absolute normalized saliency = $\mathrm{mean}_{\text{input, output}} |\frac{\partial \vec{O}}{\partial \vec{I / \mathrm{SD}[I]}}|$
  - Grid heatmap: Saliency map of an instance
  - Scatter plot: t-SNE of hidden states colored by output value
- 2E: Multi-panel: Ablation analysis of best model
  - Violin plot: MAE distribution in ablated evaluation of best model
  - Violin plot: MAE distribution in ablative re-training of best model

## Figure 3: Dynamic Structure Prediction and Analysis

- 3A: Violin plot: Difference in MAE(best model - RN) w.r.t. dataset entry, grouped by sample
- 3B: KDE plot: Structural variability distribution for experimental icSHAPE + best model prediction w.r.t. transcript ID // index of dynamicity: RMSD
- 3C: KDE plot: Mean MAE of RibonanzaNet + best model w.r.t. transcript ID // reflects prediction accuracy of dynamic structure
- 3D: Multi-panel:
  - 3D1: Flow chart: Negative control workflow 1 (sequence -> secondary structure prediction)
  - 3D2: Flow chart: Negative control workflow 2 (sequence -> in vitro icSHAPE prediction -> secondary structure prediction)
  - 3D3: Flow chart: Proposed workflow (sequence -> RNA-Seq -> in vitro icSHAPE prediction -> best model -> secondary structure prediction)
  - 3D4: Flow chart: Positive control workflow (experimental icSHAPE data -> secondary structure prediction)
- 3E: Multi-panel
  - ROC curves & PR curves: Comparison of different workflows
  - Table: F-scores, precision, recall of different workflows on test set.
- 3F: 2D structure diagrams and dot-bracket notation: Comparison of secondary structures for one example sequence predicted by different workflows (4) at different time points (4/5) // 4x4 grid

## Supplementary Figure 1: Dataset Processing, Assembly, and Characteristics

- (From Figure 1A): [selective supplementary]
- (From Figure 1B): [selective supplementary]

## Supplementary Figure 2: Model Architectures

- S2A: Diagram: H08 architecture
- S2B: Diagram: H09 architecture
- S2C: Diagram: H10 architecture
- S2D: Diagram: H11 architecture
- S2E: Diagram: H12 architecture

## Supplementary Figure 3: Residual Analysis of Best Model

- (From Figure 2B): [selective supplementary]

## Supplementary Figure 4: Performance Analysis of Best Model

- (From Figure 2C): [selective supplementary]
