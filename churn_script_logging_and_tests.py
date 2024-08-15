"""
Unit tests for the customer churn prediction project.

This script tests key functions in the churn_library.py module, including data import, 
exploratory data analysis (EDA), feature encoding, feature engineering, and model training.

Functions:
----------
- test_import: Verifies data import functionality.
- test_eda: Ensures EDA plots are generated correctly.
- test_encoder_helper: Checks proper encoding of categorical features.
- test_perform_feature_engineering: Validates train/test split and data preparation.
- test_train_models: Confirms models are trained, predictions are made, and models are saved.
"""

import os
import logging
import churn_library as cl

OUTPUT_DIR = 'images/'
TEST_EDA_DIR = os.path.join(OUTPUT_DIR, 'test_eda')
TEST_RESULTS_DIR = os.path.join(OUTPUT_DIR, 'test_results')

# Ensure the logs directory exists
if not os.path.exists('./logs'):
    os.makedirs('./logs')

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/test_data.csv")
        cl.logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        cl.logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        cl.logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    # Setup: Define the test directory and ensure it's clean before running
    # the test
    if os.path.exists(TEST_EDA_DIR):
        for filename in os.listdir(TEST_EDA_DIR):
            file_path = os.path.join(TEST_EDA_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                cl.logger.error(
                    "Failed to clean up test directory: %s. Error: %s", file_path, e)

    # Import data and run perform_eda
    try:
        df = cl.import_data("./data/test_data.csv")
        perform_eda(df, TEST_EDA_DIR)
        cl.logger.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        cl.logger.error("Testing perform_eda: FAILED with error: {err}")
        raise err

    # Verify the expected output files are created
    expected_files = [
        'cat_Gender.png',
        'cat_Education_Level.png',
        'cat_Marital_Status.png',
        'cat_Income_Category.png',
        'cat_Card_Category.png',
        'quant_Customer_Age.png',
        'quant_Dependent_count.png',
        'quant_Months_on_book.png',
        'quant_Total_Relationship_Count.png',
        'quant_Months_Inactive_12_mon.png',
        'quant_Contacts_Count_12_mon.png',
        'quant_Credit_Limit.png',
        'quant_Total_Revolving_Bal.png',
        'quant_Avg_Open_To_Buy.png',
        'quant_Total_Amt_Chng_Q4_Q1.png',
        'quant_Total_Trans_Amt.png',
        'quant_Total_Trans_Ct.png',
        'quant_Total_Ct_Chng_Q4_Q1.png',
        'quant_Avg_Utilization_Ratio.png',
        'kde_Customer_Age.png',
        'kde_Credit_Limit.png',
        'kde_Total_Revolving_Bal.png',
        'kde_Avg_Open_To_Buy.png',
        'kde_Total_Amt_Chng_Q4_Q1.png',
        'kde_Avg_Utilization_Ratio.png',
        'bivar_Credit_Limit_vs_Total_Trans_Amt.png',
        'bivar_Customer_Age_vs_Avg_Utilization_Ratio.png',
        'heatmap_correlation.png'
    ]

    missing_files = [
        f for f in expected_files if not os.path.exists(
            os.path.join(
                TEST_EDA_DIR, f))]

    if missing_files:
        cl.logger.error(
            "Testing perform_eda: The following expected files were not found: %s", missing_files)
        raise FileNotFoundError(f"Expected files not found: {missing_files}")
    else:
        cl.logger.info(
            "Testing perform_eda: All expected files were successfully created.")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df_test = cl.import_data("./data/test_data.csv")

    # Manually add the 'Churn' column based on 'Attrition_Flag'
    df_test['Churn'] = df_test['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

# Category columns for encoding
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

# Run the encoder_helper function
    df_test = encoder_helper(df_test, category_lst, response='Churn')

# Check if the new columns are added
    try:
        for category in category_lst:
            new_column_name = f'{category}_Churn'
            assert new_column_name in df_test.columns, \
                f"{new_column_name} not found in DataFrame columns"
            assert df_test[new_column_name].isnull().sum(
            ) == 0, f"{new_column_name} contains null values"

        cl.logger.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        cl.logger.error("Testing encoder_helper: FAILED with error: %s", err)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df_test = cl.import_data("./data/test_data.csv")

    # Manually add the 'Churn' column based on 'Attrition_Flag'
    df_test['Churn'] = df_test['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

# Category columns for encoding
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

# Run the encoder_helper function
    df_test = cl.encoder_helper(df_test, category_lst, response='Churn')

# Run the perform_feature_engineering function
    X_train, X_test, y_train, y_test = perform_feature_engineering(df_test)

    # Check that the function returns four outputs
    try:
        assert len([X_train, X_test, y_train, y_test]
                   ) == 4, "Function did not return four outputs"

        # Check that the sizes of X_train, X_test, y_train, y_test are correct
        assert len(X_train) == len(
            y_train), "Mismatch between X_train and y_train lengths"
        assert len(X_test) == len(
            y_test), "Mismatch between X_test and y_test lengths"

        # Check that the split proportions are correct
        total_samples = len(df_test)
        train_size = int(total_samples * 0.7)
        test_size = total_samples - train_size

        assert len(
            X_train) == train_size, f"Expected X_train length: {train_size}, got {len(X_train)}"
        assert len(
            X_test) == test_size, f"Expected X_test length: {test_size}, got {len(X_test)}"

        cl.logger.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        cl.logger.error(
            "Testing perform_feature_engineering: FAILED with error: %s", err)
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df_test = cl.import_data("./data/test_data.csv")

    # Manually add the 'Churn' column based on 'Attrition_Flag'
    df_test['Churn'] = df_test['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

# Category columns for encoding
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

# Run the encoder_helper function
    df_test = cl.encoder_helper(df_test, category_lst, response='Churn')

# Run the perform_feature_engineering function
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(df_test)

    # Run the train_models function
    try:
        y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = train_models(
            X_train, X_test, y_train, y_test, 'test_', TEST_RESULTS_DIR)

        # Check the outputs
        assert len(y_train_preds_lr) == len(
            y_train), "Mismatch in y_train_preds_lr length"
        assert len(y_train_preds_rf) == len(
            y_train), "Mismatch in y_train_preds_rf length"
        assert len(y_test_preds_lr) == len(
            y_test), "Mismatch in y_test_preds_lr length"
        assert len(y_test_preds_rf) == len(
            y_test), "Mismatch in y_test_preds_rf length"

        # Check if the models were saved correctly
        assert os.path.exists(
            './models/test_rfc_model.pkl'), "Test Random Forest model was not saved"
        assert os.path.exists(
            './models/test_logistic_model.pkl'), "Test Logistic Regression model was not saved"

        cl.logger.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        cl.logger.error("Testing train_models: FAILED with error: %s", err)
        raise err


def main():
    """
    Main function to run all unit tests for the churn library.
    """
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)


if __name__ == "__main__":
    main()
