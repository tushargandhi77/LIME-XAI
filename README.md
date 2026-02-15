# LIME-XAI

This repository demonstrates Explainable AI (XAI) using **LIME (Local Interpretable Model-Agnostic Explanations)** on two supervised learning tasks:

- Customer churn **classification** (tabular, imbalanced target)
- California housing price **regression** (tabular, continuous target)

The project is structured as notebook-first experimentation with full model building, hyperparameter tuning using Optuna, and local prediction explanations using LIME.

## Repository Structure

```text
LIME-XAI/
|-- Classification Case Study/
|   |-- classification_LIME.ipynb
|   `-- Churn_Modelling.csv
|-- Regression Case Study/
|   `-- regression_LIME.ipynb
|-- pyproject.toml
|-- uv.lock
|-- main.py
`-- README.md
```

## Project Goals

- Build strong baseline and tuned models for classification and regression.
- Show how black-box predictions can be interpreted locally with LIME.
- Compare model behavior before and after hyperparameter tuning.
- Provide an end-to-end educational reference for tabular XAI workflows.

## Environment and Dependencies

Python requirement from `pyproject.toml`: **>= 3.11**

Main libraries:

- `scikit-learn`
- `xgboost`
- `lightgbm`
- `optuna`
- `lime`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`

## Setup

### Option 1: Using `uv` (recommended)

```bash
uv sync
```

### Option 2: Using `pip`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Then launch Jupyter and run notebooks:

```bash
jupyter notebook
```

## Case Study 1: Customer Churn Classification

Notebook: `Classification Case Study/classification_LIME.ipynb`  
Dataset: `Classification Case Study/Churn_Modelling.csv`

### Problem

Predict whether a customer exits (`exited = 1`) or stays (`exited = 0`).

### Data and Preprocessing

- Initial data shape: **(10000, 14)**
- Dropped identifier columns: `RowNumber`, `CustomerId`, `Surname`
- Lowercased feature names for cleaner pipeline operations
- Missing values: none
- Duplicate rows: none
- Added engineered feature:
  - `iszerobal` = 1 if `balance == 0`, else 0
- Target distribution is imbalanced:
  - Class 0: **~79.63%**
  - Class 1: **~20.37%**

### Feature Groups

- Numerical (scaled): `creditscore`, `age`, `balance`, `estimatedsalary`
- Categorical (one-hot): `geography`, `gender`
- Passthrough: `tenure`, `numofproducts`, `hascrcard`, `isactivemember`, `iszerobal`

Preprocessing is done with a `ColumnTransformer`:

- `MinMaxScaler` for numerical columns
- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categorical columns
- `set_config(transform_output="pandas")` keeps transformed output as DataFrame

### Model

Soft-voting ensemble (`VotingClassifier`) with:

- `RandomForestClassifier(class_weight="balanced")`
- `XGBClassifier(class_weight="balanced")`
- `LGBMClassifier(class_weight="balanced")`

Train/test split: `test_size=0.2`, `random_state=42`, `stratify=y`.

### Hyperparameter Tuning

- Library: **Optuna**
- Trials: **100**
- Cross-validation: `StratifiedKFold`
- Optimization target: mean CV F1 score

Best study value observed:

- **0.6386503067484662**

Best tuned parameters include:

- `rf__n_estimators=82`
- `rf__max_depth=6`
- `rf__max_samples=0.6326428980790929`
- `xgb__learning_rate=0.07146813846933586`
- `xgb__max_depth=6`
- `xgb__n_estimators=189`
- `xgb__subsample=0.8026361163935809`
- `xgb__colsample_bynode=0.8575915574177483`
- `xgb__reg_lambda=7.554903158507428`
- `lgbm__max_depth=5`
- `lgbm__n_estimators=120`
- `lgbm__learning_rate=0.11526989570425891`
- `lgbm__subsample=0.979456684606814`
- `lgbm__reg_lambda=7.0450803247388505`

### Test Performance (Classification Report)

- Accuracy: **0.84**
- Class 1 (churn) precision: **0.61**
- Class 1 (churn) recall: **0.64**
- Class 1 F1-score: **0.62**

Interpretation:

- The model is strong on majority class detection.
- Minority class recall is moderate, indicating useful churn sensitivity with room for further improvement.

### LIME for Classification

LIME is configured with:

- `mode="classification"`
- `training_data=X_train.values`
- `feature_names=preprocessor.get_feature_names_out().tolist()`
- Categorical feature indices mapped to transformed matrix

A sample local explanation (single customer) highlights top rules such as:

- `1.00 < numofproducts <= 2.00` (negative contribution)
- `isactivemember=1` (negative contribution)
- `0.19 < age <= 0.26` (negative contribution)
- `estimatedsalary > 0.74` (positive contribution)

This demonstrates how local, instance-specific feature effects can differ from global feature assumptions.

## Case Study 2: California Housing Regression

Notebook: `Regression Case Study/regression_LIME.ipynb`

### Problem

Predict median house value (`MedHouseVal`) using California district-level features.

### Data

- Source: `fetch_california_housing(as_frame=True)`
- Shape: **(20640, 8)**
- Features:
  - `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- Split: `test_size=0.2`, `random_state=42`

### Baseline Model

`XGBRegressor` with manually chosen initial hyperparameters.

Observed baseline metrics:

- Train RMSE: **0.2740**, Train R2: **0.9438**
- Test RMSE: **0.4460**, Test R2: **0.8482**

### Hyperparameter Tuning

- Library: **Optuna**
- Trials: **50**
- Objective optimized by cross-validation score (R2)

Best tuned parameters:

- `learning_rate=0.0956733210606355`
- `n_estimators=474`
- `max_depth=7`
- `reg_lambda=86.78218250818065`
- `subsample=0.9583608345950781`

Best study value:

- **0.8554783010084526**

### Tuned Model Metrics

- Train RMSE: **0.2752**, Train R2: **0.9433**
- Test RMSE: **0.4391**, Test R2: **0.8529**

Interpretation:

- Tuning yields a measurable test improvement over the baseline.
- Train/test gap remains controlled, suggesting reasonable generalization.

### LIME for Regression

LIME is configured with:

- `mode='regression'`
- `training_data=X_train.values`
- `feature_names` from California housing columns

A sample explanation (single district) includes strong local effects such as:

- `Latitude > 37.72` (negative)
- `Longitude <= -121.81` (positive)
- `MedInc <= 2.57` (negative)
- `AveRooms <= 4.45` (negative)

This shows localized, interpretable feature contributions for one prediction value.

## Why LIME Here

LIME is useful in this repository because it:

- Works with any black-box model (`VotingClassifier`, `XGBRegressor`)
- Explains individual predictions rather than only global behavior
- Converts complex model outputs into human-readable local rules
- Supports debugging, trust, and stakeholder communication

## Current Limitations and Improvement Ideas

- Add model persistence (`joblib`) for reproducibility outside notebooks.
- Add fixed random seeds consistently for all estimators during tuning.
- Add threshold tuning/calibration for churn classification to improve minority-class recall.
- Add SHAP comparison for global + local interpretability.
- Add unit tests or notebook validation checks for CI.
- Move reusable preprocessing/training code from notebooks into `src/` modules.

## Reproducibility Notes

- Notebook outputs include concrete metrics and local explanations, but LIME outputs can vary with sampling unless random state is fixed in explainer configuration.
- Optuna tuning results may vary across runs due to stochastic search.

## Quick Start Workflow

1. Install dependencies.
2. Run `classification_LIME.ipynb` end-to-end.
3. Run `regression_LIME.ipynb` end-to-end.
4. Inspect `as_list()`, `as_pyplot_figure()`, and HTML explanation outputs for LIME in both tasks.

## License

Add your preferred license (MIT, Apache-2.0, etc.) in a `LICENSE` file.
