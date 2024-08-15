"""
churn_library.py

This module contains functions for performing exploratory data analysis (EDA),
feature engineering, model training, and evaluation for predicting customer churn.

Functions:
- import_data: Loads data from a CSV file into a DataFrame.
- perform_eda: Performs exploratory data analysis and saves visualizations.
- encoder_helper: Encodes categorical features using the target variable.
- perform_feature_engineering: Splits the data into training and testing sets.
- train_models: Trains models, saves them, and generates performance metrics.
- classification_report_image: Creates and saves classification reports as images.
- feature_importance_plot: Generates and saves feature importance plots.
"""

import os
import logging
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

OUTPUT_DIR = 'images/'
EDA_DIR = os.path.join(OUTPUT_DIR, 'eda')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        return df
    except FileNotFoundError:
        print(f"File at path {pth} not found.")
        return None


def perform_eda(df, output_dir=None):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            output_dir: The directory where the EDA images will be saved.
                        If None, uses the default EDA_DIR.

    output:
            None
    '''
    if output_dir is None:
        output_dir = EDA_DIR

    logging.info("Performing EDA...")

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    kde_plots = [
        'Customer_Age',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    bivariate_plots = [
        ('Credit_Limit', 'Total_Trans_Amt'),
        ('Customer_Age', 'Avg_Utilization_Ratio'),
    ]

    if 'Churn' not in df.columns:
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        cat_columns.append('Churn')

    for cat_column in cat_columns:
        plt.figure(figsize=(20, 10))
        df[cat_column].value_counts('normalize').plot(kind='bar')
        plt.xlabel(cat_column)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {cat_column}')
        if cat_column == 'Churn':
            plt.xticks([0, 1], ['0', '1'])
        plt.savefig(f'{output_dir}/cat_{cat_column}.png')
        plt.close()

    for quant_column in quant_columns:
        if df[quant_column].dropna().empty:
            continue
        plt.figure(figsize=(20, 10))
        df[quant_column].hist()
        plt.xlabel(quant_column)
        plt.ylabel('Count')
        plt.title(f'Distribution of {quant_column}')
        plt.savefig(f'{output_dir}/quant_{quant_column}.png')
        plt.close()

    for kde_plot in kde_plots:
        plt.figure(figsize=(20, 10))
        sns.histplot(df[kde_plot], stat='density', kde=True)
        plt.xlabel(kde_plot)
        plt.ylabel('Density')
        plt.title(f'Distribution of {kde_plot} with KDE curve')
        plt.savefig(f'{output_dir}/kde_{kde_plot}.png')
        plt.close()

    for x, y in bivariate_plots:
        plt.figure(figsize=(20, 10))
        sns.scatterplot(x=x, y=y, data=df)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'{x} vs. {y}')
        plt.savefig(f'{output_dir}/bivar_{x}_vs_{y}.png')
        plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(df[quant_columns + ['Churn']].corr(),
                annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{output_dir}/heatmap_correlation.png')
    plt.close()
    logging.info(
        "Successfully performed eda and saved figures to images folder.")


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for each categorical feature encoded
                with proportion of response
    '''
    for category in category_lst:
        category_groups = df.groupby(category)[response].mean()
        new_column_name = f'{category}_{response}'
        df[new_column_name] = df[category].map(category_groups)

    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    logging.info("Performing feature engineering...")
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following columns are missing from the DataFrame: {missing_cols}")

    if response not in df.columns:
        raise ValueError(
            f"The response column '{response}' is missing from the DataFrame.")

    X = df[keep_cols].copy()

    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    logging.info("Successfully performed feature engineering.")

    return X_train, X_test, y_train, y_test


def save_classification_report_image(y_true, y_pred, model_name, file_path):
    '''
    Helper function to create and save a classification report as an image.

    input:
            y_true: actual response values
            y_pred: predicted response values
            model_name: name of the model (e.g., 'Random Forest')
            file_path: path to save the image

    output:
            None
    '''
    plt.figure(figsize=(7, 5))
    plt.text(0.01, 1.25, f'{model_name} Train', {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_true, y_pred)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    logging.info(
        "Producing classification report for training and testing results...")
    # Define model names and their corresponding predictions
    models = [
        ('Random Forest',
         y_train,
         y_train_preds_rf,
         y_test,
         y_test_preds_rf,
         'rfc_report'),
        ('Logistic Regression',
         y_train,
         y_train_preds_lr,
         y_test,
         y_test_preds_lr,
         'lr_report')]

    for model_name, y_train_true, y_train_pred, y_test_true, y_test_pred, filename in models:
        # Save training report
        save_classification_report_image(
            y_train_true,
            y_train_pred,
            f'{model_name} Train',
            f'{RESULTS_DIR}/{filename}_train.png')
        # Save testing report
        save_classification_report_image(
            y_test_true,
            y_test_pred,
            f'{model_name} Test',
            f'{RESULTS_DIR}/{filename}_test.png')
    logging.info(
        "Successfully produced classification report for training and testing results.")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    logging.info(
        "Creating and storing the feature importances in %s", output_pth)
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    # Increase the figure height for better spacing
    plt.figure(figsize=(20, 7))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_pth)

    # Close the plot to free memory
    plt.close()
    logging.info(
        "Successfully created and stored the feature importances in %s",
        output_pth)
    return None


def train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        model_prefix='',
        output_dir='./images/results'):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              model_prefix: Prefix for the filenames when storing the trained models
                            (default is '').
              output_dir: Directory where the models and ROC curve image will be saved
                            (default is './images/results').
    output:
              y_train_preds_lr: training predictions from logistic regression
              y_train_preds_rf: training predictions from random forest
              y_test_preds_lr: test predictions from logistic regression
              y_test_preds_rf: test predictions from random forest
    '''
    logging.info("Training and storing model results...")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Grid search on RandomForestClassifier
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='newton-cg', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # Predict on training and testing data
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Print classification reports
    print("Random Forest Model - Training Classification Report")
    print(classification_report(y_train, y_train_preds_rf))
    print("Random Forest Model - Test Classification Report")
    print(classification_report(y_test, y_test_preds_rf))

    print("Logistic Regression Model - Training Classification Report")
    print(classification_report(y_train, y_train_preds_lr))
    print("Logistic Regression Model - Test Classification Report")
    print(classification_report(y_test, y_test_preds_lr))

    # Save the best model
    joblib.dump(cv_rfc.best_estimator_,
                f'./models/{model_prefix}rfc_model.pkl')
    joblib.dump(lrc, f'./models/{model_prefix}logistic_model.pkl')
    logging.info("Succesfully trained and stored model results.")

    # ROC plot
    # Increase the width to make it more rectangular
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Plot ROC curves for Random Forest and Logistic Regression
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(
        lrc, X_test, y_test, ax=ax, alpha=0.8)
    ax.set_aspect(aspect=0.5)  # Adjust the aspect ratio
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def main():
    """
    Main function that orchestrates the data processing pipeline.

    This function:
    - Imports the data from a CSV file.
    - Performs Exploratory Data Analysis (EDA) and saves the figures.
    - Encodes categorical features in the dataset.
    - Performs feature engineering to prepare the data for modeling.
    - Trains models and evaluates their performance.
    - Saves the trained models and evaluation metrics.

    No input parameters or return values.
    """
    # Import data
    df = import_data('./data/bank_data.csv')

    # Perform EDA
    if df is not None:
        perform_eda(df)

        # Encode categorical features
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        df = encoder_helper(df, category_lst)

        # Perform feature engineering
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)

        # Train models
        y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = train_models(
            X_train, X_test, y_train, y_test)

        # Generate classification reports
        classification_report_image(
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf)

        # Feature importance plot
        rfc_model = joblib.load('./models/rfc_model.pkl')
        feature_importance_plot(
            rfc_model,
            X_test,
            f'{RESULTS_DIR}/feature_importance.png')


if __name__ == "__main__":
    main()
