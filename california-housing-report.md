# When $100k Could Buy a Home: California Housing Price Prediction

**Author: Khanh Nguyen**  
**Project type:** Regression, model evaluation, and web deployment  
**Tools:** Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, statsmodels, Flask, and Gunicorn

[GitHub Repository](https://github.com/khanhqn1/californiahousepricing) · [View Notebook](https://khanhqn1.github.io/californiahousepricing/) · [Web Demo](https://californiahousepricing-c6u8.onrender.com/)

## Project overview

This project estimates median housing values for California census block groups using socioeconomic, household, and geographic features. I built a workflow that starts with data exploration and a simple baseline, compares linear and nonlinear models, examines prediction errors, and packages a trained model in a Flask application.

The strongest model in the notebook, XGBoost, achieved a held-out **R² of 0.818** and **RMSE of approximately $41,785** on the subset with recorded median values below $500,000. Compared with ordinary least squares on the same subset and test split, XGBoost reduced RMSE by approximately **36.1%**.

The main lesson was that reliable modeling depends on understanding the target, evaluation design, and deployment behavior as well as choosing an algorithm.

## Problem and objective

Housing values depend on relationships among income, location, neighborhood characteristics, and household composition. A linear model provides an interpretable starting point, but it may not capture all of these relationships.

I used this project to answer three questions:

1. How much predictive value do these features provide beyond predicting the training-set mean?
2. Does a nonlinear model improve performance over linear regression and Ridge regression?
3. How do the dataset's price ceiling and feature relationships affect the interpretation of results?

The prediction target is a **block group's median house value**, rather than the sale price of an individual home. The project is an educational model of historical housing data.

## Dataset

The notebook loads the California Housing dataset through `sklearn.datasets.fetch_california_housing`. The dataset contains **20,640 observations**, eight numeric predictors, and no missing feature values in the saved inspection output. It was derived from the **1990 U.S. census**, with one observation per census block group.

| Field | Meaning | Modeling use |
| --- | --- | --- |
| `MedInc` | Median income in the block group | Included |
| `HouseAge` | Median house age | Included |
| `AveRooms` | Average rooms per household | Included |
| `AveBedrms` | Average bedrooms per household | Removed from the feature set |
| `Population` | Block group population | Included |
| `AveOccup` | Average household occupancy | Included |
| `Latitude` | Block group latitude | Included |
| `Longitude` | Block group longitude | Included |
| `Price` | Median house value, in units of $100,000 | Target |

A prediction of `2.5` therefore represents a median value of **$250,000** in the dataset's historical dollar scale.

### Understanding the price ceiling

The target distribution has a visible ceiling near `5.0`, or $500,000. The notebook analyzes both the full dataset and a subset selected with `Price < 5.0`. This filter removes **992 observations**, leaving **19,648**.

The filtered subset avoids the ceiling-valued records, but it also excludes the most expensive block groups. It is a restricted population, not a recovery of their unknown uncapped values. Its scores cannot be compared directly with full-dataset scores as proof of overall improvement.

## Exploratory analysis and feature decisions

I checked data types, missing values, descriptive statistics, pairwise correlations, feature distributions, and geographic patterns. Scatterplots and regression plots helped examine relationships between housing values and individual predictors.

Median income had the strongest positive pairwise correlation with the target, approximately **0.688**. Average rooms and average bedrooms were strongly correlated with each other, approximately **0.848**, so I removed `AveBedrms` to reduce redundancy in the linear feature set. The same seven predictors were used for the subsequent model comparisons.

I also inspected variance inflation factors and retained both latitude and longitude because they carry location information. The notebook computes VIF without an explicit intercept, so the large reported values should be treated cautiously rather than as standard intercept-adjusted diagnostics.

## Modeling and evaluation

### Baseline and linear models

The notebook uses an **80/20 train-test split** with `random_state=42`. On the full dataset, this produces 16,512 training observations and 4,128 test observations. On the filtered subset, it produces 15,718 training observations and 3,930 test observations.

The comparison includes:

- **Mean baseline:** `DummyRegressor(strategy="mean")`, which predicts the training-set mean.
- **Ordinary least squares:** `StandardScaler` followed by `LinearRegression` in a scikit-learn pipeline.
- **Ridge regression:** The same scaling approach followed by `RidgeCV`, with 25 candidate regularization strengths from 0.001 to 1,000.

The pipelines fit scaling within training or cross-validation folds. Ridge selected an alpha of approximately **3.162**, but its test performance was effectively the same as ordinary least squares.

### XGBoost

For XGBoost, I split the filtered training partition again, reserving 20% of it for validation. This gives 12,574 fitting observations, 3,144 validation observations, and the separate 3,930-observation test set. XGBoost uses the seven features without standardization.

| Setting | Value |
| --- | --- |
| Objective | Squared error regression |
| Validation metric | RMSE |
| Learning rate | 0.05 |
| Maximum tree depth | 6 |
| Row subsampling | 0.8 |
| Column subsampling | 0.8 |
| L2 regularization (`lambda`) | 1.0 |
| Maximum boosting rounds | 5,000 |
| Early stopping patience | 50 rounds |
| Seed | 42 |

The saved run selected boosting iteration **901**, using zero-based indexing. The notebook's test predictions explicitly use the selected iteration range.

## Results

All values below come from the notebook's saved outputs. Dollar errors are calculated by multiplying MAE and RMSE by $100,000.

### Held-out test performance

| Evaluation population | Model | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Full dataset | Mean baseline | $90,607 | $114,486 | -0.0002 |
| Full dataset | OLS + scaling | $54,254 | $73,982 | 0.5823 |
| Recorded values below $500,000 | Mean baseline | $78,631 | $97,915 | Approximately 0 |
| Recorded values below $500,000 | OLS + scaling | $49,321 | $65,420 | 0.5536 |
| Recorded values below $500,000 | Ridge + scaling | $49,320 | $65,421 | 0.5536 |
| Recorded values below $500,000 | XGBoost | **$27,838** | **$41,785** | **0.8179** |

On the filtered subset, XGBoost reduced MAE by approximately **43.6%** and RMSE by **36.1%** relative to OLS. Its higher R² indicates that the nonlinear model captured substantially more variation in the held-out target values. This does not identify which particular feature interactions caused the gain; feature attribution was not implemented.

### Five-fold cross-validation

The notebook uses shuffled five-fold cross-validation with seed 42. Values after ± are standard deviations across folds, not confidence intervals.

| Population | Model | Mean RMSE ± fold SD | Mean R² ± fold SD |
| --- | --- | ---: | ---: |
| Full dataset | OLS | $73,680 ± $830 | 0.5922 ± 0.0092 |
| Recorded values below $500,000 | OLS | $64,390 ± $1,100 | 0.5602 ± 0.0113 |
| Recorded values below $500,000 | Ridge | $64,850 ± $1,320 | 0.5540 ± 0.0087 |
| Recorded values below $500,000 | XGBoost | $39,790 ± $1,060 | 0.8320 ± 0.0070 |

For XGBoost, each held-out fold is used both for early stopping and for reporting its score. Those scores can be optimistic because the fold influences iteration selection. The separate train/validation/test result is the clearer headline estimate. In addition, the cross-validation experiments span the entire analysis population, including rows from the earlier test split; these are complementary experiments rather than an additional untouched evaluation.

## Error analysis

The notebook includes residual-versus-prediction plots, residual density plots, Q-Q plots, and residuals against individual features for linear regression. It also plots the XGBoost validation RMSE across boosting rounds.

These diagnostics help assess systematic errors, departures from a simple linear relationship, and the point where additional boosting stops helping. RMSE exceeds MAE for the best model, indicating that larger errors contribute materially to the total error. The notebook does not report XGBoost error breakdowns by region or income group, so average performance does not establish equally good predictions across those groups.

## Deployment

The repository includes a pickled XGBoost booster, a Flask application, an HTML form, and a Gunicorn deployment entry point. The web demo is linked from the portfolio; the deployed service's current availability was not tested for this report.

The application accepts these features in a fixed order:

```python
["MedInc", "HouseAge", "AveRooms", "Population", "AveOccup", "Latitude", "Longitude"]
```

The `POST /predict_api` endpoint constructs an XGBoost `DMatrix` and returns a numeric prediction. The browser multiplies that prediction by 100,000 to display dollars.

One deployment detail needs attention: the notebook limits prediction to the best iteration, while `app.py` calls `Booster.predict` without that limit. Because the serialized booster can include rounds after the best iteration, deployed predictions may differ from the evaluated predictions. Passing the same `iteration_range` or exporting the selected trees would align the two paths.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `Linear_Regression_ML.ipynb` | Exploration, modeling, diagnostics, and model export |
| `docs/index.html` | Published notebook view |
| `app.py` | Flask application and prediction endpoint |
| `templates/home.html` | Input form and prediction display |
| `xgb_booster.pkl` | Serialized model |
| `requirements.txt` | Python dependencies |
| `Procfile` | Gunicorn entry point |
| `LICENSE` | Repository license |

## Running the project locally

```bash
git clone https://github.com/khanhqn1/californiahousepricing.git
cd californiahousepricing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. Open `http://127.0.0.1:5000` for the local application. To inspect or rerun the analysis, install Jupyter and SciPy if needed and open `Linear_Regression_ML.ipynb` from the repository directory. Fetching the housing dataset requires access to its download source.

These are setup instructions based on the repository, not a newly verified execution. Dependencies are unpinned, and compatibility with the pickled model depends on the environment. The notebook also retains an import-error output alongside completed results, so a clean top-to-bottom run would improve reproducibility.

## Limitations and next steps

- **Historical and aggregated data:** Results describe 1990 block group medians. They do not validate current prices or individual-property valuations.
- **Restricted target range:** The deployed model was trained on records below the target ceiling. It has no demonstrated performance for excluded high-value block groups.
- **Geographic validation:** Random splits may place nearby block groups in training and testing. A geographically grouped split would better test transfer to unseen areas.
- **Evaluation design:** Use an inner validation split for early stopping within each outer cross-validation fold, then reserve a final test partition until model selection is complete.
- **Model diagnostics:** Compare the retained feature set against all eight predictors for XGBoost, add regional error analysis, and examine feature attribution and prediction intervals.
- **Deployment consistency:** Match the evaluated iteration range, pin dependencies, and add server-side feature and range validation.

## What this project demonstrates

This project demonstrates baseline design, exploratory analysis, regression diagnostics, regularization, gradient boosting, early stopping, cross-validation, and the practical work of connecting a trained model to a web interface. Its strongest result is the measured improvement over linear regression, supported by a clear explanation of the population and evaluation method behind that result.

## Sources and reporting scope

This report was prepared from the public repository's notebook, README, application, template, dependency file, and deployment entry point. Models were not retrained for the report; reported scores are saved notebook results, and relative improvements and dollar conversions are calculated from those results.

- [Modeling notebook source](https://github.com/khanhqn1/californiahousepricing/blob/main/Linear_Regression_ML.ipynb)
- [Application source](https://github.com/khanhqn1/californiahousepricing/blob/main/app.py)
- [Web template source](https://github.com/khanhqn1/californiahousepricing/blob/main/templates/home.html)
- [Dataset source identified in the notebook](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
