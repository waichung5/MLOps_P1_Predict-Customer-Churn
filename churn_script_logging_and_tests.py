"""
Module for unit tests of churn_library,py

Project: Predict Customer Churn wit Clean Code
Author: Paul Lee
Date: 22 Apr 2022
"""

import os
import logging
import churn_library as cls
from constants import DATA_PATH, EDA_IMAGES_DIR, EDA_NAME_DICT, CAT_COLUMNS, \
    MODELS_DIR, MODELS_NAME_DICT, RESULTS_DIR, RESULTS_NAME_DICT

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(pth):
    """
    test data import - this example is completed for you to assist with the
    other test functions
    """
    try:
        dataframe = cls.import_data(pth)
    except FileNotFoundError as err:
        logging.error("Testing import_data [ERROR]: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data [ERROR]: "
                      "The file doesn't appear to have rows and columns")
        raise err

    logging.info("Testing import_data [SUCCESS]: Dataframe is "
                 "imported correctly")
    return dataframe


def test_eda(dataframe):
    """
    test perform eda function
    """

    try:
        cls.perform_eda(dataframe)
    except KeyError as err:
        logging.error("Testing perform_eda [ERROR]: "
                      "Column %d is missing", err.args[0])
        raise err

    for eda_name in EDA_NAME_DICT.values():
        try:
            filename = os.path.join(EDA_IMAGES_DIR, eda_name)
            assert os.path.isfile(filename)
        except FileNotFoundError as err:
            logging.error("Testing perform_eda [ERROR]: "
                          "File %d CANNOT be found", eda_name)
            raise err

    logging.info("Testing perform_eda [SUCCESS]: All EDA plots can be found")


def test_encoder_helper(dataframe):
    """
    test encoder helper
    """

    # load dataframe
    df_encoded = cls.encoder_helper(
        dataframe, CAT_COLUMNS, response='Churn')

    for category in CAT_COLUMNS:
        try:
            assert category in df_encoded.columns
        except AssertionError as err:
            logging.error("Testing encoder helper [ERROR]: "
                          "Category %d is missing.", category)
            raise err
    # if no error, all figures are saved
    logging.info("Testing encoder helper [SUCCESS]: "
                 "Dataframe contains all categorical features.")
    return df_encoded


def test_perform_feature_engineering(dataframe):
    """
    test perform_feature_engineering
    """

    X_train, X_test, y_train, y_test = \
        cls.perform_feature_engineering(dataframe=dataframe, response='Churn')

    # check
    try:
        assert len(X_train) == len(y_train)
    except AssertionError as err:
        logging.error("Testing feature engineering [ERROR]: "
                      "Lengths of train data sets do not match.")
        raise err

    try:
        assert len(X_test) == len(y_test)
    except AssertionError as err:
        logging.error("Testing feature engineering [ERROR]: "
                      "Lengths of test data sets do not match.")
        raise err

    logging.info("Testing feature engineering [SUCCESS]: "
                 "Train and test data sets are correctly split.")
    return X_train, X_test, y_train, y_test


def test_train_models(train_test_set):
    """
    test train_models
    """
    X_train, X_test, y_train, y_test = train_test_set

    cls.train_models(X_train=X_train,
                     X_test=X_test,
                     y_train=y_train,
                     y_test=y_test)

    # check whether models can be saved or not
    for model_name in MODELS_NAME_DICT.values():
        try:
            filename = os.path.join(MODELS_DIR, model_name)
            assert os.path.isfile(filename)
        except FileNotFoundError as err:
            logging.error("Testing train_models-save models [ERROR]: "
                          "Model %d CANNOT be found.", model_name)
            raise err

    # if no error, all models are saved correctly
    logging.info("Testing train_models-save models [SUCCESS]: "
                 "All models are saved correctly.")

    # check whether report images can be saved or not
    for result_name in RESULTS_NAME_DICT.values():
        try:
            filename = os.path.join(RESULTS_DIR, result_name)
            assert os.path.isfile(filename)
        except FileNotFoundError as err:
            logging.error("Testing train_models-save results [ERROR]: "
                          "File %d CANNOT be found.", result_name)
            raise err

    # if no error, all figures are saved correctly
    logging.info("Testing train_models-save results [SUCCESS]: "
                 "All result files are saved correctly.")


if __name__ == "__main__":
    print("\nStart testing import_data")
    dataframe_test = test_import(DATA_PATH)
    print("\nStart testing perform_eda")
    test_eda(dataframe_test)
    print("\nStart testing encoder_helper")
    dataframe_encoded = test_encoder_helper(dataframe_test)
    print("\nStart testing perform_feature_engineering")
    train_test_data = test_perform_feature_engineering(dataframe_encoded)
    print("\nStart testing train_models")
    test_train_models(train_test_data)
    print("\nUnit tests are completed, please check log file for results")
