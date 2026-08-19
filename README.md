Yes. Since you have **now added `05_robustness.py` to the existing repository**, I would update the README rather than restructure the repository or remove your existing reference/data files.

One important correction: your current GitHub repository structure is **not exactly the same** as the old README. You currently have `Scripts/`, `Full Data set .csv`, `Refrences .csv`, and `Substrate mapping .csv` at the repository root. So the README below is written to match that structure rather than claiming files are under `data/` or `src/`.

Copy-paste the following directly into `README.md`:

````markdown
# Harmonised dark-fermentation biohydrogen dataset and cross-study validation pipeline

Data and code for:

**“Cross-study generalisation fails in machine-learning models of dark fermentation: quantifying unit heterogeneity, synthetic augmentation and source-level leakage”**

**Hassan, Moustafa & Abdelkader**

This repository contains the harmonised dark-fermentation biohydrogen dataset and the complete analysis scripts used to assess machine-learning performance, cross-study generalisation, dataset heterogeneity, validation sensitivity, temperature associations, energy implications, and model robustness.

The analyses are designed to distinguish apparent predictive performance obtained from conventional random train/test splitting from genuine generalisation to previously unseen source studies.

---

## What this repository is for

Machine-learning models of dark-fermentative hydrogen production can report very high predictive performance when observations from the same source study appear in both the training and test sets.

This repository provides a reproducible framework for testing whether such performance persists when the **entire source study is held out**.

The analysis progressively examines:

1. Dataset harmonisation and conversion auditing.
2. Validation sensitivity to the way train/test sets are constructed.
3. Between-study versus within-study variation.
4. Temperature associations and pooled substrate-class effects.
5. Energy and break-even implications of thermophilic operation.
6. Algorithm, hyperparameter, category-encoding, and study-influence robustness.

The central validation principle is:

> **Observations from the same source study must not be allowed to appear simultaneously in training and testing when the objective is to assess cross-study generalisation.**

---

## Main finding

The validation framework demonstrates a strong dependence of apparent model performance on the data-splitting protocol.

| Validation protocol | Random Forest | CatBoost |
|---|---:|---:|
| Dataset duplicated with Gaussian noise, then split at random | +0.65 | +0.63 |
| Original rows split 80/20 at random | +0.44 | +0.44 |
| **Source studies held out entirely** | **−0.17** | **−0.35** |
| Mean predictor, study-grouped | — | −0.18 |

Under source-study-grouped validation, neither ensemble model demonstrates useful predictive generalisation beyond the study-grouped mean baseline.

This indicates that the high performance obtained under conventional row-wise validation is strongly influenced by **source-level structure and information leakage**, rather than representing reliable prediction for genuinely unseen studies.

Approximately **71% of the variance in harmonised hydrogen yield lies between studies**, demonstrating that source-study identity captures a substantial component of the heterogeneity in the compiled literature.

---

# Repository structure

The repository retains the original data and reference files and adds the analysis scripts under `Scripts/`.

```text
.
├── Scripts/
│   ├── common.py
│   ├── 01_audit_dataset.py
│   ├── 02_validation_ladder.py
│   ├── 03_statistics.py
│   ├── 04_energy_balance.py
│   └── 05_robustness.py
│
├── Full Data set .csv
├── Refrences .csv
├── Substrate mapping .csv
├── LICENSE
└── README.md
````

Analysis outputs are written to:

```text
results/
```

The `results/` directory is generated when the scripts are executed and contains the numerical outputs required to reproduce the reported analyses.

---

# Dataset

## `Full Data set .csv`

This is the compiled literature dataset containing the original observations used for the analysis.

The dataset contains experimental information including:

* substrate;
* microbial inoculum;
* reactor configuration/mode;
* pH;
* temperature;
* reported hydrogen yield;
* original units;
* harmonised hydrogen yield;
* conversion information;
* unit grouping;
* source reference.

The original values and units are retained for traceability.

---

## `Refrences .csv`

This file contains the bibliographic references associated with the observations in the compiled dataset.

It is retained as part of the repository so that individual observations can be traced back to their original source studies.

The reference identifier is also used as the **study-grouping variable** in the cross-study validation analyses.

---

## `Substrate mapping .csv`

This file contains the substrate-category mapping used to standardise heterogeneous substrate descriptions across the literature.

The mapping supports reproducible feature construction and prevents category definitions from being independently recreated across analysis scripts.

---

# Yield harmonisation

The source literature reports hydrogen yields using multiple denominators and unit conventions.

The harmonisation procedure separates the observations into three principal conversion groups.

### Group A — volumetric yield per mass

Values reported as:

* mL H₂ g⁻¹;
* cm³ H₂ g⁻¹;
* L H₂ g⁻¹;
* L H₂ kg⁻¹;
* or equivalent VS-based volumetric yield units.

These are converted to:

```text
dm³ H₂ g⁻¹
```

### Group B — chemically defined substrates

For chemically defined substrates reported as molar hydrogen yields:

```text
Y × 22.414 / M
```

where:

* `Y` = mol H₂ per mol substrate;
* `22.414 dm³ mol⁻¹` = molar volume of an ideal gas at STP;
* `M` = molecular mass of the named substrate.

### Group C — chemically undefined substrates

For mixed wastes, wastewaters, and other substrates for which a unique molecular mass cannot be assigned, conversion is performed on a hexose-equivalent basis:

```text
M = 180.16 g mol⁻¹
```

This convention is explicitly flagged because it introduces an additional harmonisation assumption.

### Excluded observations

Observations reported exclusively as volumetric productivity or rate measurements, such as:

```text
L H₂ L⁻¹ reactor
mmol H₂ h⁻¹ L⁻¹
```

are not converted to mass-specific yield unless the required substrate loading is available.

---

# Quality-control and audit procedure

`01_audit_dataset.py` independently reconstructs the harmonised yield from:

1. the original reported yield;
2. the original unit;
3. the conversion group;
4. the conversion divisor or molecular mass;
5. the conversion arithmetic recorded for the individual observation.

The script then compares the independently reconstructed value with the supplied harmonised value.

Quality-control flags are retained in the audited dataset rather than silently deleting observations.

Important flags include:

* `flag_arithmetic`
* `flag_unit_suspect`
* `flag_per_slurry`
* `flag_review_duplicate`
* `flag_mw_basis`
* `flag_above_thauer`
* `flag_exact_duplicate`
* `flag_pH_derived`
* `flag_T_derived`
* `recommended_for_modelling`

The modelling dataset contains **224 observations from 82 studies** after application of the predefined quality-control criteria.

The six largest studies contribute a substantial fraction of the available observations, which is one reason repeated study-grouped validation is used throughout the analysis.

---

# Analysis pipeline

## Step 01 — Dataset audit

### `Scripts/01_audit_dataset.py`

This script:

* audits the original dataset;
* independently recomputes yield conversions;
* identifies inconsistencies;
* applies the predefined quality flags;
* generates the modelling dataset;
* records the audit summary.

Main outputs include:

```text
Full_Data_set_AUDITED.csv
dataset_modelling_224.csv
results/01_audit_summary.csv
```

---

# Step 02 — Validation ladder

## `Scripts/02_validation_ladder.py`

This is the primary cross-study validation analysis.

The same modelling dataset, feature set, model definitions, and hyperparameters are evaluated under progressively stricter validation protocols.

The analysis includes:

### 1. Synthetic-noise augmentation

The dataset is duplicated with Gaussian perturbations and subsequently divided using a conventional random split.

This tests how near-duplicate observations can inflate apparent predictive performance.

### 2. Conventional row-wise random split

The original observations are divided into training and test sets without enforcing study-level separation.

This represents the type of validation commonly used in small compiled experimental datasets.

### 3. Study-grouped holdout

Entire source studies are assigned to either training or test partitions.

No observation from a held-out study can appear in the training set.

This is the key test of **cross-study generalisation**.

### Models

The validation ladder evaluates:

* Random Forest;
* CatBoost;
* mean predictor;
* linear baseline where applicable.

Performance is reported over repeated resamples rather than from a single split.

Main outputs include:

```text
results/02_ladder_scores.csv
results/02_ladder_summary.csv
results/02_error_metrics.csv
```

The grouped analysis shows that strong performance under row-wise validation does not persist when the source study is completely withheld.

---

# Step 03 — Statistical analysis

## `Scripts/03_statistics.py`

This script quantifies the statistical structure underlying the validation results.

The analyses include:

* between-study and within-study variance;
* intraclass correlation;
* random-intercept mixed-effects modelling;
* grouped learning curves;
* temperature meta-regression;
* pH effects;
* pooled hydrogen yields by substrate class.

Main outputs include:

```text
results/03_variance_components.csv
results/03_learning_curve.csv
results/03_metaregression.csv
results/03_pooled_yields.csv
```

The reported temperature association is approximately:

```text
+0.0018 dm³ H₂ g⁻¹ VS °C⁻¹
95% CI: 0.0005–0.0032
p = 0.008
```

The temperature coefficient is interpreted as a statistical association rather than a causal experimental temperature effect.

After accounting for study identity, pH does not show a detectable independent association in the fitted model.

---

# Step 04 — Energy balance

## `Scripts/04_energy_balance.py`

This script links the temperature association estimated in Step 03 to the heat requirement associated with increasing reactor temperature.

The analysis compares:

```text
37 °C → 55 °C
```

and calculates:

* hydrogen-energy recovery;
* reactor heating duty;
* net energy balance;
* yield uplift required to offset heating;
* break-even solids loading;
* sensitivity to heat recovery.

The temperature coefficient from `03_metaregression.csv` is interpreted on the `log1p(y)` scale.

Therefore, the model-implied raw-scale yield at 55 °C is obtained by back-transformation rather than by simply multiplying the coefficient by the temperature difference.

The analysis explicitly treats the resulting yield change as:

> **model-implied yield difference associated with a 37 → 55 °C temperature shift**

rather than as a causal experimental temperature effect.

Main outputs:

```text
results/04_energy_balance.csv
results/04_breakeven_uplift.csv
results/04_breakeven_loading.csv
```

The heating threshold is optimistic because mixing and pumping requirements are excluded.

---

# Step 05 — Robustness analysis

## `Scripts/05_robustness.py`

Step 05 extends the validation analysis to test whether the main conclusions depend on the selected algorithm, hyperparameters, category representation, or influential source studies.

The response variable is explicitly defined as:

```text
y
```

with the machine-learning response transformed as:

```text
ly = log1p(y)
```

The study identifier is:

```text
ref
```

so that source-study structure remains explicit throughout the analysis.

---

## A. Algorithm robustness

The following models are evaluated under repeated study-grouped holdout:

* Random Forest;
* CatBoost;
* XGBoost;
* Ridge regression;
* mean predictor.

The same grouped validation framework is applied across algorithms.

This tests whether poor cross-study performance is specific to a particular ensemble algorithm or represents a broader limitation of the compiled dataset.

Output:

```text
results/05_robustness_algorithms.csv
```

---

## B. Random Forest hyperparameter robustness

The analysis compares:

1. default Random Forest;
2. tuned Random Forest.

Hyperparameter tuning is performed **only inside the training partition**.

The outer test partition remains completely untouched until final evaluation.

This prevents hyperparameter optimisation from leaking information from the held-out studies into model selection.

The tuning analysis uses grouped inner resampling and repeated outer grouped holdout.

Output:

```text
results/05_robustness_tuning.csv
```

---

## C. Alternative category encodings

The Random Forest is evaluated using three representations of substrate and process categories:

### Paper mapping

The predefined categorical mapping used in the primary analysis.

### Raw literature strings

The original categorical strings retained from the literature compilation.

### Coarse substrate grouping

A reduced substrate representation intended to test whether the conclusion depends on the granularity of substrate classification.

The same study-grouped validation protocol is used for all encoding variants.

Output:

```text
results/05_robustness_encoding.csv
```

---

## D. Leave-one-study-out influence

The analysis evaluates whether the main grouped-validation result is dominated by a small number of large studies.

The full modelling dataset is first evaluated.

The six largest source studies are then removed individually and the grouped validation is repeated.

For each analysis, the following are reported:

* study removed;
* number of observations removed;
* remaining observations;
* remaining number of studies;
* median grouped R²;
* 95% empirical interval.

Output:

```text
results/05_robustness_influence.csv
```

This analysis provides a direct sensitivity test for study-level leverage.

---

# Why study-grouped validation matters

The dataset is not a collection of independent observations generated under a common experimental design.

Each published study can contribute multiple observations sharing:

* analytical procedures;
* substrate preparation;
* inoculum source;
* reactor configuration;
* measurement methodology;
* experimental conditions;
* reporting conventions;
* laboratory-specific effects.

Consequently, randomly splitting rows can place highly related observations from the same study into both training and testing partitions.

A model can therefore learn study-specific structure and still appear highly predictive.

Study-grouped validation removes this source of leakage by ensuring that the test set represents genuinely unseen source studies.

The repository therefore treats **study-level grouping as the relevant validation unit for assessing cross-study generalisation**.

---

# Reproducibility

All stochastic analyses use fixed random seeds.

Repeated grouped holdout is used rather than relying on a single train/test split.

The robustness analysis further checks:

* algorithm choice;
* hyperparameter tuning;
* category encoding;
* influential source studies.

The goal is not to identify the single model with the highest apparent R².

The goal is to determine whether the predictive conclusion remains valid when the validation protocol is made appropriate for a heterogeneous literature-derived dataset.

---

# Installation

Python **3.11 or newer** is recommended.

Clone the repository:

```bash
git clone https://github.com/Hergelicious/ml-biohydrogen-dark-fermentation-dataset.git
cd ml-biohydrogen-dark-fermentation-dataset
```

Create and activate a virtual environment:

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the analysis

Run the scripts in order:

```bash
python Scripts/01_audit_dataset.py
python Scripts/02_validation_ladder.py
python Scripts/03_statistics.py
python Scripts/04_energy_balance.py
python Scripts/05_robustness.py
```

If a project-level `run_all.sh` is available, the complete pipeline can alternatively be executed using:

```bash
./run_all.sh
```

The scripts should be run sequentially because later analyses use outputs generated by earlier steps.

In particular:

```text
01 → 02 → 03 → 04 → 05
```

---

# Reusing the pipeline on another compiled dataset

The validation framework can be adapted to other literature-derived machine-learning datasets.

At minimum, a dataset should contain:

1. a response variable;
2. predictors;
3. a source-study identifier.

The critical requirement is the source-study identifier.

The grouping variable should represent the highest level of dependence that must remain entirely within either the training or test partition.

For the present dataset:

```text
TARGET = y
GROUP  = ref
```

and the machine-learning response is:

```text
log1p(y)
```

If a compiled dataset has no source-study identifier, genuine cross-study validation cannot be performed without reconstructing that provenance information.

---

# Recommended minimum standard for compiled-data machine learning

The analyses in this repository motivate the following minimum reporting practices for machine-learning studies based on published experimental observations:

1. **Retain the original unit for every observation.**
2. **Publish the conversion arithmetic for every harmonised observation.**
3. **Identify the source study explicitly and use it as the grouping variable where appropriate.**
4. **Prevent observations from the same source study appearing in both training and test sets when assessing cross-study generalisation.**
5. **Report performance over repeated resamples rather than relying on a single split.**
6. **Include a mean predictor and a simple statistical baseline.**
7. **Perform hyperparameter tuning exclusively within the training data.**
8. **Test sensitivity to alternative feature/category representations.**
9. **Assess whether large source studies disproportionately influence model performance.**
10. **Avoid treating synthetic perturbations of existing observations as independent experimental information.**
11. **Distinguish predictive association from causal experimental effects.**
12. **Publish the dataset, validation protocol, and analysis code so that the reported performance can be independently inspected.**

---

# Interpretation

A high R² obtained from a random row-wise split should not automatically be interpreted as evidence that a model can predict hydrogen yield for a new study.

For a heterogeneous literature-derived dataset, the relevant question is:

> **Can the model predict observations from a source study that was not represented during training?**

This repository shows that the answer can differ dramatically from the answer obtained using conventional row-wise validation.

The distinction is particularly important for small compiled datasets in which a relatively small number of studies contribute many observations.

---

# Citation

If you use this dataset, code, or validation framework, please cite the associated article:

**Hassan, H. Hassan Submission in process 

*Cross-study generalisation fails in machine-learning models of dark fermentation: quantifying unit heterogeneity, synthetic augmentation and source-level leakage.*

A `CITATION.cff` file can be added to this repository to provide machine-readable citation metadata.

---

# Licence

**Data:** CC BY 4.0

**Code:** MIT

See the repository licence files for the applicable terms.

