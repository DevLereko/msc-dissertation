# Explainable Loan Default Prediction Framework

Dissertation implementation using the PKDD '99 Czech Banking dataset
(Berka, 1999). Predicts loan default from leakage-controlled behavioural
features and explains predictions with SHAP.

## Repository structure
```
data/raw/      # Original PKDD '99 .asc files (not versioned)
data/processed/# Intermediate and analytical datasets
src/           # Pipeline scripts (01-12)
results/       # Figures, tables, artifacts
```

## Setup
```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline (in order)
```bash
python src/01_data_loading.py
python src/02_dataset_construction.py
python src/04_feature_engineering.py
python src/03_eda.py
python src/05_leakage_validation.py
python src/06_data_splitting.py
python src/07_model_training.py
python src/08_evaluation.py
python src/09_calibration.py
python src/10_xai_analysis.py
python src/11_stability_analysis.py
python src/12_computational_feasibility.py
```

## Data source
The task description references the classic sources
(relational.fit.cvut.cz / sorry.vse.cz), which were unreachable from the
working environment. All 8 canonical `.asc` files were retrieved from the
public GitHub mirror `jlacko/berka-dataset` which reproduces the original
PKDD '99 financial dataset byte-for-byte.