# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to predict customer churn for a bank using a combination of machine learning models, including Random Forest and Logistic Regression.

## Files and data description
The root directory of the project contains the following key files and directories:
- data/: Contains the datasets used in the project. For example:

  - bank_data.csv: The main dataset used for training and testing the models.
  - test_data.csv: A smaller dataset used for testing the functionality of the code.

- models/: Stores the trained models in .pkl format after running the training scripts.

  - rfc_model.pkl: The trained Random Forest model.
  - logistic_model.pkl: The trained Logistic Regression model.

- logs/: Contains log files that record the events that occur during the execution of the scripts.

  - churn_library.log: Logs related to the execution of the main script and testing.

- images/: Stores all the images generated from the EDA and model evaluation.

  - eda/: Contains images generated during the Exploratory Data Analysis (EDA), such as histograms, KDE plots, and scatter plots.
  - results/: Contains images of the ROC curves and feature importance plots generated during model evaluation.

```
├── Guide.ipynb
├── README.md
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   ├── bank_data.csv
│   └── test_data.csv
├── images
│   ├── eda
│   │   ├── bivar_Credit_Limit_vs_Total_Trans_Amt.png
│   │   ├── bivar_Customer_Age_vs_Avg_Utilization_Ratio.png
│   │   ├── cat_Card_Category.png
│   │   ├── cat_Churn.png
│   │   ├── cat_Education_Level.png
│   │   ├── cat_Gender.png
│   │   ├── cat_Income_Category.png
│   │   ├── cat_Marital_Status.png
│   │   ├── heatmap_correlation.png
│   │   ├── kde_Avg_Open_To_Buy.png
│   │   ├── kde_Avg_Utilization_Ratio.png
│   │   ├── kde_Credit_Limit.png
│   │   ├── kde_Customer_Age.png
│   │   ├── kde_Total_Amt_Chng_Q4_Q1.png
│   │   ├── kde_Total_Revolving_Bal.png
│   │   ├── quant_Avg_Open_To_Buy.png
│   │   ├── quant_Avg_Utilization_Ratio.png
│   │   ├── quant_Contacts_Count_12_mon.png
│   │   ├── quant_Credit_Limit.png
│   │   ├── quant_Customer_Age.png
│   │   ├── quant_Dependent_count.png
│   │   ├── quant_Months_Inactive_12_mon.png
│   │   ├── quant_Months_on_book.png
│   │   ├── quant_Total_Amt_Chng_Q4_Q1.png
│   │   ├── quant_Total_Ct_Chng_Q4_Q1.png
│   │   ├── quant_Total_Relationship_Count.png
│   │   ├── quant_Total_Revolving_Bal.png
│   │   ├── quant_Total_Trans_Amt.png
│   │   └── quant_Total_Trans_Ct.png
│   ├── results
│   │   ├── feature_importance.png
│   │   ├── lr_report_test.png
│   │   ├── lr_report_train.png
│   │   ├── rfc_report_test.png
│   │   ├── rfc_report_train.png
│   │   └── roc_curves.png
│   ├── test_eda
│   │   ├── bivar_Credit_Limit_vs_Total_Trans_Amt.png
│   │   ├── bivar_Customer_Age_vs_Avg_Utilization_Ratio.png
│   │   ├── cat_Card_Category.png
│   │   ├── cat_Churn.png
│   │   ├── cat_Education_Level.png
│   │   ├── cat_Gender.png
│   │   ├── cat_Income_Category.png
│   │   ├── cat_Marital_Status.png
│   │   ├── heatmap_correlation.png
│   │   ├── kde_Avg_Open_To_Buy.png
│   │   ├── kde_Avg_Utilization_Ratio.png
│   │   ├── kde_Credit_Limit.png
│   │   ├── kde_Customer_Age.png
│   │   ├── kde_Total_Amt_Chng_Q4_Q1.png
│   │   ├── kde_Total_Revolving_Bal.png
│   │   ├── quant_Avg_Open_To_Buy.png
│   │   ├── quant_Avg_Utilization_Ratio.png
│   │   ├── quant_Contacts_Count_12_mon.png
│   │   ├── quant_Credit_Limit.png
│   │   ├── quant_Customer_Age.png
│   │   ├── quant_Dependent_count.png
│   │   ├── quant_Months_Inactive_12_mon.png
│   │   ├── quant_Months_on_book.png
│   │   ├── quant_Total_Amt_Chng_Q4_Q1.png
│   │   ├── quant_Total_Ct_Chng_Q4_Q1.png
│   │   ├── quant_Total_Relationship_Count.png
│   │   ├── quant_Total_Revolving_Bal.png
│   │   ├── quant_Total_Trans_Amt.png
│   │   └── quant_Total_Trans_Ct.png
│   └── test_results
│       └── roc_curves.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   ├── rfc_model.pkl
│   ├── test_logistic_model.pkl
│   └── test_rfc_model.pkl
└── requirements_py3.10.txt
```


## Running Files
To run the scripts in this project, follow these steps:
1. Setup the Environment
    ```
    python -m venv .venv
    source .venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements_py3.10
    ```

2. Running the Main Script
The churn_library.py script contains the main pipeline of the project, including data import, EDA, feature engineering, and model training. To run the script:
    ```
    python churn_library.py
    ```
When you run this script, the following will happen:

* The data will be imported from data/bank_data.csv.
* EDA will be performed, and figures will be saved to the images/eda directory.
* Feature engineering will be applied to prepare the data for modeling.
* The models will be trained, and predictions will be made.
* The trained models will be saved in the models/ directory.
* ROC curves and feature importance plots will be saved in the images/results directory.

3. Running Tests
    The churn_script_logging_and_tests.py script contains unit tests to ensure that all functions in the main script are working correctly. Run the tests as follows:
    ```
    python churn_script_logging_and_tests.py
    ```
This will execute the unit tests, and the results will be logged in logs/churn_library.log.