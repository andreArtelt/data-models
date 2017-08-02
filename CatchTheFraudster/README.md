# Coursera Data Science Community Project - Catch the Fraudster
This folder contains code for working with the **"Catch the Fraudster" dataset**.

## About
The dataset contains **4349 samples** each described by **12 variables**.

| Variables   |      Explanation      |  Data Type |
|----------|:-------------:|:------:|
| Fraud Instance | Response | Binary |
|                |          |         |
| Damaged Item | Predictor | Binary |
| Item Not Available | Predictor | Binary |
| Item Not In Stock | Predictor | Binary |
| Product Care Plan | Predictor | Binary |
| Claim Amount | Predictor | Continues |
| Registered Online | Predictor | Categorical |
| Age Group | Predictor | Continues |
| Marital Status | Predictor | Categorical |
| Owns a Vehicle | Predictor | Categorical |
| Accommodation Type | Predictor | Categorical |
| Height (cms) | Predictor | Continues |

As it turns out the **most useful predictors/features** are:
- Damaged Item
- Item Not Avaiable
- Item Not In.Stock
- Product Care Plan

## Implementation
The R code (*notebook*) implements a comprehensive **data analysis** (**plots** of predictors, releations, etc.). Feature selection is done by using a **decision tree** and **stepwise logistic regression** (*directions: forward and backward*).

The python code implements **10-fold cross validation** for a **decision tree** and **logistic regression** model. Evaluation is done by *F1-Score*.

Both models achieve a *score of 1.0*, thus the **problem** can be considered as **"solved"**.

### Run the code
- Run the R notebook: ``ipython notebook Exploration.ipynb``
- Run the python code: ``python model.py``


### Requirements
- python 2.7
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [pandas](https://github.com/pandas-dev/pandas)
- R
- [rpart](https://cran.r-project.org/web/packages/rpart/index.html)
- [caret](https://cran.r-project.org/web/packages/caret/index.html)
