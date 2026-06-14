# ML Biohydrogen Dark Fermentation Dataset

**Version:** v1.0 (dataset used in the submitted manuscript)

This repository contains a standardized dataset of **246 experimental observations** on biohydrogen production via waste-fed dark fermentation, compiled from **88 independent peer-reviewed studies (2002–2025)**. All hydrogen yield values have been harmonized to a single standard unit to support machine learning modelling and cross-study comparison.

---

## Associated Manuscript

H. H. Hassan, A. M. Abdelkader,
*Ensemble Machine Learning for Cross-Study Prediction and Optimization of Biohydrogen Yield from Waste-Fed Dark Fermentation*,
*Applied Energy*, [Year], DOI: [to be updated upon acceptance]

---

## Repository Contents

- **`full_dataset.csv`**
  Main dataset containing 246 experimental observations of biohydrogen yields and associated process variables, with unit group classification and conversion notes for each row.

- **`references_full_dataset.csv`**
  Complete list of literature sources corresponding to each dataset entry, linked via Reference_ID.

- **`substrate_general_mapping.csv`**
  Mapping of raw substrate names to the generalized substrate categories used during ML preprocessing.

---

## Dataset Structure

Each row in `full_dataset.csv` represents one experimental observation. Key columns include:

| Column | Description |
|---|---|
| `Substrate` | Raw substrate name as reported in source study |
| `Substrate_Category` | Generalized category (see `substrate_general_mapping.csv`) |
| `Inoculum` | Microbial culture or inoculum type |
| `Reactor_Config` | Reactor type and operational mode |
| `pH` | Operating pH |
| `Temperature_C` | Operating temperature (°C) |
| `H2_Yield_dm3_g` | Harmonized hydrogen yield (dm³ H₂ g⁻¹ substrate or VS) |
| `Unit_Group` | Conversion group (A, B, or C) |
| `Reporting_Basis` | VS or TS basis of the original reported yield |
| `Conversion_Notes` | Arithmetic applied to each row for reproducibility |
| `Reference_ID` | Linked to `references_full_dataset.csv` |

---

## Unit Standardization

### Standard Unit

> **dm³ H₂ g⁻¹ substrate (or VS)**
> All conversions use Standard Temperature and Pressure (STP: 0 °C, 101.325 kPa), where **1 mol H₂ = 22.414 dm³**.

Yields in the source literature were reported using heterogeneous units. All values were harmonized to dm³ H₂ g⁻¹ using the three-group conversion framework below.

---

### Molecular Weights Used

| Substrate | MW (g mol⁻¹) |
|---|---|
| Glucose / hexose equivalent | 180.16 |
| Sucrose / Lactose / Cellobiose / Maltose | 342.30 |
| Glycerol | 92.09 |
| C5 pentoses (xylose, arabinose) | 150.13 |

---

### Conversion Groups

**Group A — Direct volumetric/mass yields (n ≈ 124)**
Yields already expressed per unit mass of substrate or VS, requiring only scale conversion (e.g., mL → dm³, cm³ → dm³, mmol × 22.414 ÷ 1000).

**Group B — Molar yields from pure substrates (n = 57)**
Yields expressed as mol H₂ mol⁻¹ of a known pure substrate, converted using:
```
Y (dm³ g⁻¹) = Y_molar × 22.414 / MW_substrate
```

**Group C — Molar yields from complex substrates (n = 63)**
Yields expressed as mol H₂ mol⁻¹ glucose equivalent for heterogeneous substrates (food waste, lignocellulosic biomass, wastewaters, agro-industrial residues), converted using glucose MW (180.16 g mol⁻¹) as a conservative basis.

---

### Example Conversions

| Original Value | Unit | Group | Result |
|---|---|---|---|
| 10.02 mol H₂ mol⁻¹ sucrose | Group B | → (10.02 × 22.414) / 342.30 | 0.656 dm³ g⁻¹ |
| 76 cm³ H₂ g⁻¹ VS | Group A | → 76 / 1000 | 0.076 dm³ g⁻¹ |
| 1.33 mmol H₂ g⁻¹ COD | Group A | → (1.33 × 22.414) / 1000 | 0.030 dm³ g⁻¹ |
| 13.13 N-L H₂ kg⁻¹ VS | Group A | → 13.13 / 1000 | 0.013 dm³ g⁻¹ |
| 2.1 mol H₂ mol⁻¹ glucose-eq. | Group C | → (2.1 × 22.414) / 180.16 | 0.261 dm³ g⁻¹ |

---

### VS vs. TS Basis

Group A entries are expressed on a VS basis; Groups B and C on a total dry weight (TS) basis. This heterogeneity was retained as tree-based ensemble models capture substrate-category-specific yield distributions through the categorical substrate feature. Researchers requiring strict basis unification should apply VS/TS correction factors from the primary literature for Group A entries.

---

## Relationship to Supplementary Information

The full dataset with conversion notes is provided in **Supplementary Table S3** of the manuscript. The reference list is in **Supplementary Table S4**, and substrate category mappings in **Supplementary Table S5**. This repository provides the machine-readable version to facilitate reuse and reproducibility.

---

## Citation

If you use this dataset or code in your work, please cite:

> H. H. Hassan, A. M. Abdelkader,
> *Ensemble Machine Learning for Cross-Study Prediction and Optimization of Biohydrogen Yield from Waste-Fed Dark Fermentation*,
> *Applied Energy*, [Year], DOI: [to be updated upon acceptance]

---

## License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You are free to share and adapt the material for any purpose, provided appropriate credit is given.
