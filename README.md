# Harmonised Dark-Fermentation Biohydrogen Dataset and Cross-Study Validation Pipeline

Data and code for *"Cross-study generalisation fails in machine-learning models of dark fermentation: quantifying unit heterogeneity, synthetic augmentation and source-level leakage"* (Hassan, Moustafa & Abdelkader).

Everything reported in the paper is reproduced by `./run_all.sh` (or executing the individual Python scripts in sequence). No figures are generated here; the scripts write the underlying analysis tables and execution logs directly to `results/` as CSV and log files.

---

## What this repository is for

Machine-learning models of dark-fermentative hydrogen yield are usually trained on yields compiled from published experiments, routinely reporting $R^2$ above $0.9$. This repository contains the dataset and the pipeline needed to evaluate what those performance metrics actually measure.

The central experiment holds the data, features, encoding, and hyperparameters fixed, varying only how the test set is drawn. It finds:

| Validation protocol | Random Forest | CatBoost |
| --- | --- | --- |
| Dataset duplicated with Gaussian noise, then split at random | +0.65 | +0.63 |
| Original rows split 80/20 at random | +0.44 | +0.44 |
| **Source studies held out entirely** | **−0.17** | **−0.35** |
| Mean predictor, study-grouped | — | −0.18 |

Under study-level holdout, neither ensemble model beats predicting the overall mean. The reason is a fundamental property of the data rather than the algorithm: **71% of the variance in harmonised yield lies *between* studies**, which is precisely the component a grouped holdout removes.

---

## Repository layout

```
data/
  Full_Data_set.csv               the compilation as submitted (261 records, 84 sources)
  Full_Data_set_AUDITED.csv       + recomputed conversions, quality flags    [generated]
  dataset_modelling_224.csv       the modelling subset with features attached [generated]

scripts/
  common.py                       shared configuration, mappings, feature definitions
  01_audit_dataset.py             recompute every conversion; flag; write modelling set
  02_validation_ladder.py         the three validation protocols + baselines
  03_statistics.py                variance components, learning curve, meta-regression
  04_energy_balance.py            reactor heat balance coupled to the temperature effect
  05_robustness.py                algorithm comparisons, hyperparameter tuning, sensitivity
  06_basis_sensitivity.py         sensitivity analysis across unit conventions & groups
  07_target_basis_sensitivity.py  validation ladder restricted by strict target mass basis
  08_what_transfers.py            rank transfer, classification, interval coverage & ICC

results/                          all output CSV tables and execution logs    [generated]
run_all.sh                        runs scripts 01 -> 08 in sequence

```

`common.py` holds category mappings and constants in a central location, ensuring hyperparameter definitions and preprocessing logic remain identical across all scripts.

---

## Installation and use

Python 3.11 or newer is required.

```bash
git clone https://github.com/Hergelicious/ml-biohydrogen-dark-fermentation-dataset.git
cd ml-biohydrogen-dark-fermentation-dataset
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run_all.sh                                          # or execute scripts sequentially

```

Runtime is approximately 5–10 minutes. Execution time is dominated by resampling loops (`02_validation_ladder.py` and `05_robustness.py`). All random seeds are fixed in `common.py`, guaranteeing reproducible outputs.

---

## The dataset

`data/Full_Data_set.csv` — 261 records from 84 published sources (2002–2025).

| Column | Contents |
| --- | --- |
| `Substrate`, `Microbial Inoculum`, `Reactor / Mode` | As reported by the original source |
| `pH`, `Temp (°C)` | As reported, including ranges and operational tolerances |
| `Original Yield`, `Original Unit` | Value and unit string exactly as published |
| `dm³ H₂/g` | Harmonised yield ($\text{dm}^3\ \text{H}_2\ \text{g}^{-1}$ substrate at STP) |
| `Conversion Calculation & Notes` | Arithmetic used per record |
| `Unit Group` | Group A, B, C, or Excluded |
| `Reference` | Source study (primary grouping variable for cross-validation) |

Twenty-six distinct unit conventions appear in the source literature. Harmonisation follows four rules:

* **Group A** — Volumetric per mass ($\text{mL}$, $\text{cm}^3$, or $\text{L}\ \text{H}_2$ per $\text{g}$ or $\text{kg}\ \text{VS}$): scaled directly to $\text{dm}^3\ \text{H}_2\ \text{g}^{-1}$.
* **Group B** — Molar yields for chemically defined substrates: $Y \times 22.414 / M$, where $22.414\ \text{dm}^3\ \text{mol}^{-1}$ is the molar volume at STP and $M$ is the molar mass of the named substrate.
* **Group C** — Molar yields for chemically undefined substrates (wastewaters, complex mixed wastes): converted on a hexose-equivalent basis ($M = 180.16\ \text{g}\ \text{mol}^{-1}$).
* **Excluded** — Volumetric productivities ($\text{L}\ \text{H}_2\ \text{L}^{-1}\ \text{reactor}$) or rates ($\text{mmol}\ \text{H}_2\ \text{h}^{-1}\ \text{L}^{-1}$), which cannot be made mass-specific without unstated loading metrics.

`01_audit_dataset.py` re-derives every harmonised value from the original yield and conversion notes: **256 of 257 convertible records agree**, with the single discrepancy corrected in the audited file.

### Quality flags

Written by `01_audit_dataset.py` and retained in `data/Full_Data_set_AUDITED.csv`:

| Flag | Records | Meaning |
| --- | --- | --- |
| `flag_arithmetic` | 1 | Harmonised value disagreed with stated conversion by $>2\%$ |
| `flag_unit_suspect` | 1 | Duplicates adjacent record with a factor-1000 discrepancy |
| `flag_per_slurry` | 16 | Yield per volume of slurry, not per mass of volatile solids |
| `flag_review_duplicate` | 12 | Observation already present under primary study |
| `flag_mw_basis` | 1 | Xylose on a $\text{C}_6$ basis while other xylose records use $\text{C}_5$ |
| `flag_above_thauer` | 8 | Yield above $4\ \text{mol}\ \text{H}_2\ \text{mol}^{-1}$ hexose equivalent |
| `flag_exact_duplicate` | 4 | Exact duplicate of an earlier record in same source |
| `flag_pH_derived` / `flag_T_derived` | 15 | Operational parameters parsed from range or text |
| `recommended_for_modelling` | 224 | Passes quality criteria and has usable yield |

**The modelling set contains 224 observations from 82 studies.** 47 studies contribute a single observation; the six largest contribute 38% of all rows.

---

## Detailed pipeline structure

### Execution script overview

* `01_audit_dataset.py`: Recomputes unit conversions, audits arithmetic integrity, applies quality flags, and writes the filtered modelling subset (`dataset_modelling_224.csv`).
* `02_validation_ladder.py`: Evaluates performance across the three validation protocols (noise-augmented random split, row-wise split, and study-grouped holdout) across multiple model implementations.
* `03_statistics.py`: Calculates intraclass correlation coefficients (ICC), evaluates learning curves under grouped holdout, performs random-effects meta-regression, and calculates pooled yields.
* `04_energy_balance.py`: Couples the empirical meta-regression temperature coefficient ($37^\circ\text{C} \rightarrow 55^\circ\text{C}$) to a reactor thermal balance to calculate break-even solids loadings.
* `05_robustness.py`: Evaluates sensitivity to alternative algorithms (e.g., CatBoost, XGBoost, Ridge), feature encoding schemes, hyperparameter configurations, and high-leverage observations.
* `06_basis_sensitivity.py`: Tests model behavior and validation ladders across different original unit classifications.
* `07_target_basis_sensitivity.py`: Evaluates whether the grouped performance collapse persists when restricting the dataset to mathematically strict target mass bases (e.g., volatile solids only).
* `08_what_transfers.py`: Assesses secondary model capabilities under grouped holdout, including Spearman rank correlation, coarse yield classification into tertiles, prediction interval calibration, error distribution against training centroids, and feature-level ICC.

---

## Generated outputs (`results/`)

Running the pipeline populates `results/` with the following output files and logs:

```
results/
├── 01_audit.log
├── 01_audit_summary.csv
├── 02_error_metrics.csv
├── 02_ladder_scores.csv
├── 02_ladder_summary.csv
├── 02_validation.log
├── 03_learning_curve.csv
├── 03_metaregression.csv
├── 03_pooled_yields.csv
├── 03_statistics.log
├── 03_temperature_effect.csv
├── 03_variance_components.csv
├── 04_breakeven_loading.csv
├── 04_breakeven_uplift.csv
├── 04_energy.log
├── 04_energy_balance.csv
├── 05_robustness.log
├── 05_robustness_algorithms.csv
├── 05_robustness_encoding.csv
├── 05_robustness_influence.csv
├── 05_robustness_tuning.csv
├── 06_basis.log
├── 06_basis_ladder.csv
├── 07_basis_composition.csv
├── 07_basis_sensitivity.csv
├── 07_classification.csv
├── 07_error_vs_centroid.csv
├── 08_interval_coverage.csv
├── 08_learning_curve_slope.csv
├── 08_predictor_icc.csv
├── 08_rank_transfer.csv
└── 08_transfers.log

```

---

## Minimum standard for compiled-data models

Based on findings from this pipeline:

1. **Explicit Conversions**: Record original units for every observation and publish exact conversion arithmetic per row rather than stating a generic methodology.
2. **Grouped Holdouts**: Always group by source study (`Reference`) when performing train/test splits, and explicitly report the grouping variable in the methods.
3. **Resampling Distributions**: Report evaluation metrics as distributions across repeated resamples rather than relying on a single train/test split.
4. **Baseline Comparisons**: Benchmarks must include a grouped mean predictor and linear baseline. Algorithms that cannot outperform a naive mean predictor under grouped holdout provide no predictive power.
5. **Synthetic Augmentation Restrictions**: Avoid generating synthetic samples via noise addition or row duplication, as this introduces severe data leakage under random splitting.
6. **Reproducible Pipelines**: Publish full data and code scripts to ensure validation protocols can be independently audited.

---

## License

* **Data**: Creative Commons Attribution 4.0 International (`LICENSE`)
* **Code**: MIT License (`LICENSE-CODE`)

Please cite the associated manuscript when using the dataset or code in academic work (see `CITATION.cff`).
