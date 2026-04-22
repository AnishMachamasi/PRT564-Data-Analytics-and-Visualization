# PRT564 — Predicting Chronic Condition Prevalence by Age in Australia
 
**Unit:** PRT564 Data Analytics & Visualisation  
**Assessment:** Group Project (Assessment 2)  
**Group:** 12 — CDU Sydney Campus
 
## Team Members
 
| Name | Student ID |
|------|------------|
| Ashish Dhakal | S395996 |
| Anish Machamasi | S389151 |
| Roshan Mahato | S390410 |
| Sadan Magar | S394031 |
 
---
 
## Project Overview
 
This project predicts age-specific **Obesity Class 1 prevalence** across Australia using the ABS National Health Survey (NHS) 2022 dataset, enriched with PBS pharmaceutical data.
 
**Research Question:** Can we predict age-specific Obesity Class 1 prevalence using NHS health, lifestyle and socioeconomic indicators — enriched with PBS pharmaceutical data?
 
---
 
## Data Sources
 
- **NHS 2022 (ABS TableBuilder)** — 10 CSV files covering clinical, behavioural, and socioeconomic variables
- **PBS Data (Heterogeneous Source)** — Tables 20 & 21: prescription scripts per person and concession co-payment data by age
---
 
## Pipeline Summary
 
1. **Download** — 10 NHS CSVs from ABS TableBuilder
2. **Parse** — Auto-detect matrix vs long format
3. **Clean** — Remove title rows, fix headers, handle nulls
4. **Merge** — Join all sources by age key
5. **Normalise** — Count ÷ population = comparable rates
6. **Filter** — Adults only (age ≥ 16)
**Final dataset:** 84 observations (age 16–99), 128 variables, 0 missing values.
 
---
 
## Models Used
 
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.102 | 0.168 | −3.39 |
| Ridge (α=1.0) | 0.086 | 0.136 | −1.85 |
| Lasso (α=0.001) | 0.076 | 0.115 | −1.05 |
| **Random Forest ✓** | **0.047** | **0.060** | **0.435** |
 
All models evaluated with **10-fold cross-validation**.
 
---
 
## Key Findings
 
- Obesity peaks at ages **45–65**, then declines (survivorship bias)
- **Random Forest** significantly outperforms all linear models (p < 0.001)
- **SEIFA disadvantage** and **smoking rate** are the strongest lifestyle predictors
- Linear models produced negative R² — confirming a non-linear age-obesity relationship
---
 
## Repository Structure
 
```
├── outputs_regression_dataset/               # Transformed Dataset
├── outputs_transformed_data/             # Raw NHS CSV files and PBS tables
├── notebooks/          # Exploratory analysis notebooks
├── ./
│   ├── 01_parse.py     # Data parsing
│   ├── 02_preprocess.py# Cleaning, merging, normalisation
│   ├── 03_regression.py# Model training and cross-validation
│   └── 04_evaluation.py# Metrics and statistical tests
├── outputs/            # Plots, results, figures
├── Presentation/       # contains presentation file for this project
└── README.md
```
 
---
 
## Requirements
 
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```
 
---
 
## How to Run
 
```bash
python 02_preprocess.py
python 03_regression.py
python 04_evaluation.py
```
 