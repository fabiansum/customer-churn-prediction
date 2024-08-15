import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import RocCurveDisplay, classification_report

import logging

import os
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    output_dir = 'image/eda'
    os.makedirs(output_dir, exist_ok=True)
    
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
        ('Credit_Limit','Total_Trans_Amt'),
        ('Customer_Age','Avg_Utilization_Ratio'),
    ]
    
    if 'Churn' not in df.columns:
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        cat_columns.append('Churn')
        
        
    for cat_column in cat_columns:
        plt.figure(figsize=(20,10)) 
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
        plt.figure(figsize=(20,10))
        df[quant_column].hist()
        plt.xlabel(quant_column)
        plt.ylabel('Count')        
        plt.title(f'Distribution of {quant_column}')
        plt.savefig(f'{output_dir}/quant_{quant_column}.png')
        plt.close()
        
    for kde_plot in kde_plots:
        plt.figure(figsize=(20,10))
        sns.histplot(df[kde_plot], stat='density', kde=True)
        plt.xlabel(kde_plot)
        plt.ylabel('Density')    
        plt.title(f'Distribution of {kde_plot} with KDE curve')
        plt.savefig(f'{output_dir}/kde_{kde_plot}.png')
        plt.close()
    
    for x, y in bivariate_plots:
        plt.figure(figsize=(20,10))
        sns.scatterplot(x=x, y=y, data=df)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'{x} vs. {y}')
        plt.savefig(f'{output_dir}/bivar_{x}_vs_{y}.png')
        plt.close()
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(df[quant_columns + ['Churn']].corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{output_dir}/heatmap_correlation.png')
    plt.close()


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for each categorical feature encoded with proportion of response
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
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    
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
        raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")
    
    if response not in df.columns:
        raise ValueError(f"The response column '{response}' is missing from the DataFrame.")

    X = df[keep_cols].copy()
    
    y = df[response]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

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
    pass


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
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    

def main():
    pass

if __name__ == "__main__":
    main()