Biohydrogen Yield Dataset – Standardized Compilation

This repository contains a standardized dataset of 250 experimental data points on biohydrogen production, compiled from diverse literature sources.  
All hydrogen yield values have been unified to a single standard unit to allow consistent comparison and support data-driven modeling.

---

Repository Contents

- `FULL_DATA_SET_cvs.csv`  
  → Main dataset with 250 samples of biohydrogen yields from different substrates, microbial cultures, and experimental conditions.

- `Refrences_full_dataset.csv`  
  → Literature references corresponding to each sample in the dataset.

- `substrate_general_mapping.csv`  
  → A cleaned mapping of raw substrate names to generalized categories used in preprocessing.

---

Unit Standardization for Biohydrogen Yield

Hydrogen yield values were reported using various units across studies.  
To ensure consistency and support meaningful comparison and modeling, all yield values were converted to a unified unit:

Standard Unit: `dm³ H₂ / g substrate`

Purpose of Standardization
To standardize the dataset, all values were converted to `dm³ H₂ / g substrate` using appropriate assumptions and molecular weights where needed.

Example Conversions

1. From `mol H₂ / mol substrate` to `dm³ H₂ / g substrate`
   Given: 10.02 mol H₂ / mol sucrose  
   MW of sucrose = 342.3 g/mol  
   → `dm³ H₂ / g = (10.02 × 22.4) / 342.3 ≈ 0.656 dm³ H₂ / g sucrose`

2. From `cm³ H₂ / g VS` to `dm³ H₂ / g VS`
   Given: 76 cm³ H₂ / g VS  
   → `dm³ H₂ = 76 / 1000 = 0.076 dm³ H₂ / g VS`

3. From `mmol H₂ / g COD` to `dm³ H₂ / g COD` 
   Given: 1.33 mmol H₂ / g COD  
   → `dm³ H₂ = (1.33 / 1000) × 22.4 ≈ 0.0298 dm³ H₂ / g COD`

4. From `N-L H₂ / kg VS` to `dm³ H₂ / g VS`
   Given: 13.13 N-L / kg  
   → `dm³ H₂ / g = 13.13 / 1000 = 0.01313 dm³ H₂ / g VS`

STP (Standard Temperature and Pressure) was assumed: 1 mol H₂ = 22.4 dm³

---

Citation

If you use this dataset in your work, please cite the associated paper:  
[Title, Authors, Journal, DOI/link – to be added after publication]
