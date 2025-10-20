# Lithium-ion Battery State of Health Estimation using AutoML Frameworks

## Overview
This repository supports the study:

**"Lithium-ion battery state of health estimation using automated machine learning (AutoML): performance comparison"**  
**Authors:** Hasibe Candan Kadem, Onur Kadem

The study compares seven open-source AutoML frameworks for estimating the State of Health (SoH) of lithium-ion batteries using features derived from real-world degradation data. Experiments are conducted using the Oxford Battery Degradation Dataset with a Leave-One-Cell-Out (LOCO) nested cross-validation strategy.

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

This study presents a reproducible benchmark for comparing seven open-source AutoML frameworks in predicting the state of health (SoH) of lithium-ion batteries. Using the Oxford Battery Degradation Dataset, features such as Equivalent Full Cycles, IC peak amplitude, and mean cell temperature were extracted to capture electrochemical and thermal aging behavior. A key methodological contribution is the use of leave-one-cell-out nested cross-validation, which ensures generalizability across cells with varying degradation profiles. All frameworks were evaluated under identical preprocessing, metrics, and runtime constraints. Results show that AutoML tools like FLAML, TPOT, AutoGluon, and H2O AutoML can provide highly accurate predictions without manual tuning, while frameworks such as PyCaret and EvalML offer better interpretability. LazyPredict proved useful for rapid model screening. The findings support AutoML as a scalable and effective approach for battery SoH estimation. Future directions include integrating domain knowledge, uncertainty quantification, and edge deployment for real-time battery management systems.

