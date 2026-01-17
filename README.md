# ML Biohydrogen Dark Fermentation Dataset

**Version:** v1.0 (dataset used in the submitted manuscript)

This repository contains a standardized dataset of 250 experimental observations on biohydrogen production in waste-fed dark fermentation systems, compiled from diverse peer-reviewed literature sources.  
All hydrogen yield values have been unified to a single standard unit to enable consistent comparison and support data-driven and machine-learning-based modeling.

---

## Repository Contents

- **`full_dataset.csv`**  
  Main dataset containing 250 samples of biohydrogen yields and associated process variables (substrate, inoculum, reactor configuration, pH, temperature, etc.).

- **`references_full_dataset.csv`**  
  Complete list of literature sources corresponding to each dataset entry.

- **`substrate_general_mapping.csv`**  
  Mapping of raw substrate names reported in the literature to generalized substrate categories used during preprocessing.

---

## Dataset Structure

Each row in `full_dataset.csv` represents one experimental observation.  
Key columns include:

- Substrate  
- Inoculum / microbial culture  
- Reactor configuration  
- pH  
- Temperature (°C)  
- Hydrogen yield (dm³ H₂ g⁻¹ substrate)  
- Reference_ID (linked to `references_full_dataset.csv`)

This structure ensures full traceability of each data point to its original literature source.

---

## Unit Standardization for Biohydrogen Yield

Hydrogen yield values were originally reported using heterogeneous units across studies.  
To ensure consistency and support meaningful comparison and modeling, all values were converted to a unified standard unit:

**Standard unit:** `dm³ H₂ g⁻¹ substrate`

### Purpose of Standardization

All hydrogen yield values were converted to `dm³ H₂ g⁻¹ substrate` using appropriate molecular weights and standard assumptions, enabling direct comparison across studies and compatibility with machine learning workflows.

### Example Conversions

1. **From `mol H₂ mol⁻¹ substrate` to `dm³ H₂ g⁻¹ substrate`**  
   Given: 10.02 mol H₂ mol⁻¹ sucrose  
   Molecular weight of sucrose = 342.3 g mol⁻¹  
   → `(10.02 × 22.4) / 342.3 ≈ 0.656 dm³ H₂ g⁻¹ sucrose`

2. **From `cm³ H₂ g⁻¹ VS` to `dm³ H₂ g⁻¹ VS`**  
   Given: 76 cm³ H₂ g⁻¹ VS  
   → `76 / 1000 = 0.076 dm³ H₂ g⁻¹ VS`

3. **From `mmol H₂ g⁻¹ COD` to `dm³ H₂ g⁻¹ COD`**  
   Given: 1.33 mmol H₂ g⁻¹ COD  
   → `(1.33 / 1000) × 22.4 ≈ 0.0298 dm³ H₂ g⁻¹ COD`

4. **From `N-L H₂ kg⁻¹ VS` to `dm³ H₂ g⁻¹ VS`**  
   Given: 13.13 N-L H₂ kg⁻¹ VS  
   → `13.13 / 1000 = 0.01313 dm³ H₂ g⁻¹ VS`

Standard Temperature and Pressure (STP) was assumed:  
**1 mol H₂ = 22.4 dm³**

---

## Relationship to Supplementary Information

The same dataset is reproduced in Supplementary Table S1 of the associated manuscript for archival purposes.  
This GitHub repository provides a machine-readable version of the dataset to facilitate reuse, transparency, and reproducibility.

---

## Citation

If you use this dataset in your work, please cite the associated paper:

H. H. Hassan *et al.*,  
*Machine Learning Framework for Optimizing Biohydrogen Yields in Waste-Fed Dark Fermentation Systems*,  
Journal, Year, DOI (to be added after publication).

---

## License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
