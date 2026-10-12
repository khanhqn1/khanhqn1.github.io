---
layout: default
title: Customer Churn Project Report
permalink: /customer-churn-report/
---

# Customer Churn Prediction with an Artificial Neural Network

**Author: Khanh Nguyen**  
**Project type:** Binary classification and deep learning  
**Tools:** Python, TensorFlow/Keras, pandas, NumPy, scikit-learn, and Matplotlib

[GitHub Repository](https://github.com/khanhqn1/churnmodelingann) · [View Notebook](https://khanhqn1.github.io/churnmodelingann/) · [Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling/data)

## Project overview

This project predicts whether a bank customer will leave using customer demographics, account characteristics, and activity indicators. I built a feedforward neural network and worked through the steps needed to use it with tabular data: feature selection, categorical encoding, scaling, training, regularization, and evaluation.

The saved run achieved **85.35% accuracy** on a 2,000-customer test set. Looking beyond accuracy, the model achieved **73.15% precision** and **40.20% recall** for customers who churned at a probability threshold of 0.5.

These results show the central trade-off in the current model: its positive predictions are relatively precise, but it misses a substantial proportion of customers who leave. The project therefore demonstrates both a neural-network workflow and the importance of choosing metrics that match the intended business use.

## Problem and objective

Customer retention teams have limited time and resources for outreach. A churn model could help identify customers for investigation or a retention campaign before they leave.

The objective here is to learn a binary mapping from customer attributes to the `Exited` label:

- `0`: The customer did not exit.
- `1`: The customer exited.

The network produces a score between zero and one. The notebook classifies a customer as likely to churn when that score is **at least 0.5**. These scores are model outputs; their calibration as real-world probabilities was not evaluated.

The dataset provides an exit label rather than a defined future prediction window. The project does not establish a validated prediction of churn within a particular number of days.

## Dataset and features

The notebook reads `Churn_Modelling.csv`, linked from the portfolio through Kaggle. The saved split and test outputs account for **10,000 customers**. The source table contains 14 columns: three identifiers, ten predictors, and the target.

| Column | Meaning | Modeling use |
| --- | --- | --- |
| `RowNumber` | Row index | Removed |
| `CustomerId` | Customer identifier | Removed |
| `Surname` | Customer surname | Removed |
| `CreditScore` | Credit score | Included |
| `Geography` | Customer country | One-hot encoded |
| `Gender` | Recorded gender | One-hot encoded |
| `Age` | Customer age | Included |
| `Tenure` | Length of the customer relationship | Included |
| `Balance` | Account balance | Included |
| `NumOfProducts` | Number of bank products | Included |
| `HasCrCard` | Credit card indicator | Included |
| `IsActiveMember` | Active membership indicator | Included |
| `EstimatedSalary` | Estimated salary | Included |
| `Exited` | Whether the customer exited | Target |

Removing row numbers, customer IDs, and surnames keeps the feature set focused on reusable customer attributes rather than record-specific identifiers. The notebook does not include a comprehensive missing-value or duplicate audit, so such checks remain a useful extension.

## Preprocessing

### Categorical encoding

I used `pandas.get_dummies` with `drop_first=True` to encode geography and gender. The resulting indicator columns are `Germany`, `Spain`, and `Male`, with France and Female represented by the omitted reference categories in the saved data.

After replacing the two original categorical columns, the model has **11 input features**: eight remaining predictors and three encoded indicators. Dropping a reference category removes redundant columns, though it is not required to avoid a linear-model singularity in this neural network.

### Training and test split

The notebook uses an **80/20 random split** with `random_state=42`:

| Partition | Customers | Role |
| --- | ---: | --- |
| Training pool | 8,000 | Scaling and subsequent fitting/validation split |
| Test set | 2,000 | Final classification evaluation |

During training, Keras reserves 33% of the training pool for validation. This produces approximately **5,360 fitting customers** and **2,640 validation customers**. The test set remains separate from the training loop.

### Feature scaling

`StandardScaler` is fitted on the 8,000-customer training pool and applied to the test set. This prevents test observations from influencing the fitted scaling parameters.

There is a smaller validation-design issue: scaling occurs before Keras separates the validation portion, so validation features contribute to the scaler's statistics. A stricter workflow would create the fitting and validation partitions first and fit the scaler only on the fitting partition. Encoding is also performed before the initial split; a fitted encoder would provide a clearer reusable preprocessing interface.

## Neural network architecture

I implemented a Keras `Sequential` model with three hidden Dense layers and a sigmoid output. The first Dense layer has 11 units; the input itself consists of 11 scaled features.

| Layer | Configuration | Purpose |
| --- | --- | --- |
| Input | 11 scaled features | Customer attributes |
| Dense 1 | 11 units, ReLU | Learn an initial nonlinear representation |
| Dense 2 | 6 units, ReLU | Combine learned signals |
| Dropout 1 | Rate 0.2 | Regularize the preceding representation |
| Dense 3 | 6 units, ReLU | Learn additional combinations |
| Dropout 2 | Rate 0.3 | Further regularization |
| Output | 1 unit, sigmoid | Produce a binary-classification score |

The network is compiled with the **Adam optimizer**, **binary cross-entropy loss**, and **accuracy** as a tracked metric. Dropout randomly deactivates a fraction of activations during training to reduce dependence on particular pathways.

This is a compact demonstration architecture. The notebook does not compare alternative networks or establish that an ANN is the best model for this dataset.

## Training and overfitting controls

| Training setting | Value |
| --- | --- |
| Batch size | 10 |
| Maximum epochs | 1,000 |
| Validation share of training pool | 0.33 |
| Early stopping monitor | Validation loss |
| Minimum qualifying improvement | 0.0001 |
| Early stopping patience | 20 epochs |
| Restore best weights | False |

The saved run stopped at **epoch 79**, well before the maximum of 1,000. The notebook plots training and validation accuracy and loss across epochs to examine learning progress.

Because `restore_best_weights=False`, evaluation uses the weights from the final completed epoch rather than automatically restoring the best validation-loss weights. Enabling restoration would make the evaluated network correspond more directly to the early-stopping selection criterion.

The learning-curve legends say “test,” but those curves use `val_accuracy` and `val_loss`. They describe the validation partition, not the separate 2,000-customer test set.

## Results

The following confusion matrix is saved in the notebook. Rows represent actual outcomes, and columns represent predictions at the 0.5 threshold.

| Actual outcome | Predicted stayed | Predicted churned | Total |
| --- | ---: | ---: | ---: |
| Stayed | 1,549 true negatives | 58 false positives | 1,607 |
| Churned | 235 false negatives | 158 true positives | 393 |
| Total | 1,784 | 216 | 2,000 |

### Classification metrics

Accuracy is reported directly by the notebook. The other metrics below are calculated from its saved confusion matrix, treating churn as the positive class.

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Accuracy | **85.35%** | 1,707 of 2,000 customers classified correctly |
| Churn precision | **73.15%** | 158 of 216 predicted churners actually churned |
| Churn recall | **40.20%** | 158 of 393 actual churners were identified |
| Churn F1 score | **0.519** | Balance between precision and recall |
| Specificity | **96.39%** | 1,549 of 1,607 non-churners identified correctly |
| Always predict “stayed” baseline | **80.35% accuracy** | Test-set comparison calculated from class counts |

The model improves accuracy by **5.00 percentage points** over predicting that every test customer stays. This baseline is a calculated reference, not a separately trained model in the notebook.

### What these results mean

Only **19.65%** of test customers churned, so accuracy is strongly influenced by the larger non-churn class. The model correctly identifies most customers who stay, but misses **235 of 393 churners**, or approximately **59.8%**.

If the threshold were used to choose customers for outreach, the saved test result would select 216 customers, including 158 who actually churned and 58 who did not. That illustrates a relatively selective list, but it also shows how many potential retention opportunities would be missed.

Lowering the threshold could improve recall while increasing false positives. The appropriate threshold depends on outreach costs, the value of retaining a customer, and how effective the intervention is. This project does not include those inputs or measure the effect of retention campaigns, so it does not claim a revenue or retention improvement.

## Repository contents

| File | Purpose |
| --- | --- |
| `Churn_Modeling_ANN.ipynb` | Preprocessing, network training, learning curves, and evaluation |
| `docs/index.html` | Published notebook view |

The repository snapshot reviewed for this report does not contain the input CSV, a saved model, a saved scaler, a dependency manifest, or a deployed inference application. The current deliverable is a modeling notebook and its published HTML view.

## Running the notebook

1. Clone the repository and create a Python environment.
2. Install the analysis dependencies.
3. Download `Churn_Modelling.csv` from the linked Kaggle dataset and place it in the repository directory, or update the notebook's file path.
4. Open the notebook and execute its cells in order.

```bash
git clone https://github.com/khanhqn1/churnmodelingann.git
cd churnmodelingann
python -m venv .venv
source .venv/bin/activate
pip install tensorflow pandas numpy scikit-learn matplotlib jupyter
jupyter notebook Churn_Modeling_ANN.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate`. The saved notebook reports **TensorFlow 2.19.0**. These instructions are based on the source and were not newly executed for this report. A compatible, pinned environment would make reproduction more predictable.

The split seed is fixed, but the notebook does not explicitly fix TensorFlow's random seed. Weight initialization, dropout, and training order can therefore produce different results across runs.

## Limitations and next steps

- **Recall and imbalance:** Track precision, recall, F1, PR-AUC, and ROC-AUC alongside accuracy. Compare class weighting and threshold selection using validation data.
- **Model comparison:** Add a majority-class baseline, logistic regression, and tree-based models on consistent splits before choosing a final architecture.
- **Validation discipline:** Use stratified partitions, fit preprocessing only on fitting data, and restore the best validation-loss weights.
- **Reproducibility:** Pin dependencies, set random seeds, audit missing values and duplicates, and save the encoder, scaler, and network together.
- **Interpretation:** Add feature attribution and error analysis. The current notebook does not establish which customer attributes drive predictions.
- **Prediction timing:** Define a future churn window and verify that every feature is available at the intended prediction time.
- **Operational usefulness:** Evaluate calibration, performance across customer groups, and retention-campaign outcomes before treating predictions as a business decision system.

These are proposed extensions; they are not implemented results in the current notebook.

## What this project demonstrates

This project demonstrates converting mixed customer data into numeric inputs, training a small feedforward network, applying dropout and early stopping, and interpreting a confusion matrix. The key finding is that **85.35% accuracy does not by itself establish strong churn detection**. Understanding precision and recall makes the result more useful and provides a clear direction for the next iteration.

## Sources and reporting scope

This report was prepared from the public notebook and repository structure, together with the dataset link on the portfolio. The model was not retrained. Accuracy and confusion-matrix counts are saved notebook results; precision, recall, F1, specificity, and the majority-class reference are derived from those counts.

- [Notebook source](https://github.com/khanhqn1/churnmodelingann/blob/main/Churn_Modeling_ANN.ipynb)
- [Published notebook](https://khanhqn1.github.io/churnmodelingann/)
- [Dataset linked from the portfolio](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling/data)
