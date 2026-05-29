# Bank Fraud Detection System using Machine Learning (DECISION TREE)

This repository contains a binary classification system developed to identify fraudulent banking transactions. The project involves end-to-end data processing, handling heavily imbalanced financial logs, training a Decision Tree algorithm, and verifying the results using specific confusion matrix metrics.

---

## Model Classification Results

The classifier was evaluated on a test dataset split, achieving an accuracy score of 99.9704%. The exact transaction classification distribution from the confusion matrix breakdown is documented below:

* True Positives (Actual Fraud, Predicted Fraud): 1,906,107 transactions
* True Negatives (Actual Non-Fraud, Predicted Non-Fraud): 2,114 transactions
* False Positives (Actual Non-Fraud, Predicted Fraud): 321 transactions
* False Negatives (Actual Fraud, Predicted Non-Fraud): 244 transactions

---

## Technical Project Steps

The notebook implements a structured machine learning pipeline across the following data stages:

1. Data Exploration: Loads the transaction dataset into Pandas dataframes to evaluate shape properties, look at descriptions, check unique value splits, and scan for any structural null values.
2. Feature Separation: Splits the columns into predictor variables (X) and the target classification column (y).
3. Data Splitting: Utilizes Scikit-Learn utilities to split the rows into distinct training sets and testing slices.
4. Model Fitting: Initializes and trains a Decision Tree Classifier (`tree.fit`) directly on the training data arrays.
5. Metrics Evaluation: Compares predicted target labels against true test flags to extract final accuracy percentages and exact confusion metrics.

---


## Tech Stack and Requirements

* Execution Language: Python 3
* Data Handling: NumPy, Pandas
* Data Visualization: Matplotlib, Seaborn
* Predictive Modeling: Scikit-Learn (`train_test_split`, Decision Tree algorithms)

---

## Running the Notebook

### Runtime Environment Setup
Make sure your active working environment contains the source fraud dataset file referenced in the code initialization steps.

### Dependency Installation
Before executing the script cells locally, install the core data science libraries using your terminal manager:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
