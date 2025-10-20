# Lithium-ion Battery State of Health Estimation using AutoML Frameworks

## Overview
This repository supports the study:

**"Lithium-ion battery state of health estimation using automated machine learning (AutoML): performance comparison"**  
**Authors:** Hasibe Candan Kadem, Onur Kadem

The study benchmarks seven open-source AutoML frameworks for lithium-ion battery State-of-Health (SoH) estimation using physics-informed features derived from real-world degradation data. Experiments are conducted on two public datasets—the Oxford Battery Degradation Dataset (Cells 1–8) and the Mohtat et al. (2021, University of Michigan) Li-NMC pouch-cell dataset (Cells 01, 03, 10, 11, 12)—under a nested Leave-One-Cell-Out (LOCO) cross-validation protocol to assess battery-level generalization. The feature set includes, for example, equivalent full cycles (EFC), IC-based normalized peak magnitude, and temperature statistics, harmonized across datasets.

---

## Dataset Information
Dataset1
- **Name:** Oxford Battery Degradation Dataset  
- **Source:** [Oxford Research Archive](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)  
- **Format:** `.mat` files (converted to CSV)  
- **Cells:** Cell1 through Cell8  
- **Description:** Contains voltage, current, and temperature measurements under controlled laboratory conditions.
  
Dataset2
- **Name:** UofM Pouch Cell Voltage and Expansion Cyclic Aging Dataset (Mohtat et al., 2021)
- **Source:** [University of Michigan – Deep Blue Data](https://deepblue.lib.umich.edu/data/concern/data_sets/5d86p0488)
- **Format:** `.csv` files
- **Cells:** Cell01, Cell03, Cell10, Cell11 and Cell12
- **Description:** Voltage, current, temperature and expansion measurements under controlled conditions; characterization tests at specific intervals. Measurements were recorded at 10 s intervals.

Preprocessed feature files are stored in the `FeatureEngineering/` directory.

---

## AutoML Frameworks Included

The following AutoML frameworks are evaluated under identical runtime and validation conditions:

- FLAML  
- TPOT  
- H2O AutoML  
- AutoGluon  
- PyCaret  
- EvalML  
- LazyPredict  

Each framework is limited to a 900-second runtime budget and executed with default internal hyperparameter spaces.

---

## Folder Structure

battery-SoH-AutoML/
├── FLAML/
│ └── flaml_900s.py
├── TPOT/
│ └── tpot_900s.py
├── H2O_AutoML/
│ └── h2o_900s.py
├── AutoGluon/
│ └── autogluon_900s.py
├── PyCaret/
│ └── pycaret_900s.py
├── EvalML/
│ └── evalml_900s.py
├── LazyPredict/
│ └── lazypredict_900s.py
├── FeatureEngineering/
│ ├── Oxford_Battery_Degradation_Dataset_1.mat # Original raw dataset (MAT file)
│ ├── cell*_features.csv # Extracted feature files (Output file including selected features for each cell)
│ ├── feature_eng.m # MATLAB script for EFC, temp, and SoH computation (driver code)
│ └── IC_analysis.m # MATLAB script for ICA analysis feature extraction (MATLAB function)
├── info.txt
└── README.md

---

## Feature Engineering

Raw cycling data from each battery cell was processed using **MATLAB** to extract relevant diagnostic features. This preprocessing stage included:

- **Equivalent Full Cycles (EFC):**  The total amount of charge throughput normalized by the nominal (rated) capacity of the battery.
- **Incremental Capacity Peak Amplitude (dQ/dV):** The magnitude of the most prominent peak from Incremental Capacity Analysis (ICA), capturing electrochemical degradation.
- **Mean Operating Temperature:** Average temperature over each charge-discharge cycle.
- **Computed State of Health (SoH):** Ratio of current capacity to initial capacity (normalized).

The resulting features were exported as `.csv` files and stored in the `FeatureEngineering/` directory for use by Python-based AutoML models.


---

## Methodology

1. Extract features from raw sensor data.
2. Use Leave-One-Cell-Out cross-validation (LOCO).
3. Run each AutoML framework on the same training/test folds.
4. Evaluate using RMSE, MAE, and R² metrics.
5. Use Python 3.8.10; GPU acceleration disabled.

---

### Install dependencies:
Run an AutoML pipeline:

python FLAML/flaml_900s.py
python TPOT/tpot_900s.py
...


Requirements
Python 3.8.10

numpy, pandas, scikit-learn

flaml, tpot, h2o, autogluon, pycaret, evalml, lazypredict

matplotlib, seaborn, scipy


## Conclusion

This study delivers a reproducible benchmark of seven open-source AutoML frameworks for lithium-ion battery SoH prediction across two public datasets—Oxford and Mohtat/U-Michigan Li-NMC—using a physics-informed feature set (Equivalent Full Cycles, IC-based Normalized_Peak, mean temperature) and a nested leave-one-cell-out protocol with harmonized preprocessing, common metrics, and fixed runtime budgets. Results show that FLAML and TPOT achieve the lowest errors, AutoGluon and H2O AutoML exhibit strong generalization and automation, EvalML and PyCaret provide enhanced interpretability, and LazyPredict is effective for rapid baseline screening. SHAP analyses are consistent with domain intuition: Normalized_Peak is a monotonic positive driver of SoH, EFC is a net negative usage signal (steeper early life, milder later), and meanT exerts a smaller, typically negative effect. Compared with non-AutoML baselines used in this work (polynomial, GPR, LSTM), the best AutoML configurations attain comparable or lower errors under identical budgets, suggesting practical viability for embedded BMS/EV monitoring given low inference costs. Limitations include reliance on laboratory datasets and manually engineered features; future work will tighten physics priors, quantify uncertainty, explore automated feature synthesis, and optimize edge deployment.
