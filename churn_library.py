# library doc string
"""
Module for predicting customer churn

Project: Predict Customer Churn wit Clean Code
Author: Paul Lee
Date: 22 Apr 2022
"""

# import libraries

import logging
import os
#import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
from constants import DATA_PATH, EDA_IMAGES_DIR, EDA_NAME_DICT, CAT_COLUMNS,\
    KEEP_COLS, MODELS_DIR, MODELS_NAME_DICT, RESULTS_DIR, RESULTS_NAME_DICT

sns.set()


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    try:
        dataframe = pd.read_csv(pth)
        # prepare dataframe with 'Churn'
        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return dataframe
    except FileNotFoundError as err:
        logging.error("ERROR: We were not able to find that file")
        raise err


def perform_eda(dataframe):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """

    # save figures based on items in dict IMAGE_NAME_DICT
    for key, value in EDA_NAME_DICT.items():
        plt.figure(figsize=(20, 10))
        if key == 'Churn':
            # Churn histogram plot
            dataframe['Churn'].hist()
        elif key == 'Customer_Age':
            # Customer Age histogram plot
            dataframe['Customer_Age'].hist()
        elif key == 'Marital_Status':
            # Marital status counts plot
            dataframe.Marital_Status.\
                value_counts('normalize').plot(kind='bar')
        elif key == 'Total_Trans_Ct':
            # Total transaction distribution plot
            sns.distplot(dataframe['Total_Trans_Ct'])
        elif key == 'Heatmap':
            # Correlations heatmap plot
            sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r',
                        linewidths=2)
        plt.savefig(os.path.join(EDA_IMAGES_DIR, value),
                    bbox_inches='tight')
        plt.close()


def encoder_helper(dataframe, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from
    the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """

    dataframe_encoded = dataframe.copy()

    for cat in category_lst:
        data_list = []
        groups = dataframe.groupby(cat).mean()['Churn']

        for val in dataframe[cat]:
            data_list.append(groups.loc[val])

        if response:
            dataframe_encoded[cat + '_' + response] = data_list
        else:
            dataframe_encoded[cat] = data_list

    return dataframe_encoded


def perform_feature_engineering(dataframe, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    df_encoded = encoder_helper(dataframe=dataframe,
                                category_lst=CAT_COLUMNS,
                                response=response)

    y = df_encoded['Churn']

    X = pd.DataFrame()
    X[KEEP_COLS] = df_encoded[KEEP_COLS]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores
    report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """

    # results from random forest
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, RESULTS_NAME_DICT['random_forest']))
    plt.close()

    # results from logistic regression
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, RESULTS_NAME_DICT['logistic']))
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importance
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth, bbox_inches='tight')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    print("\nStart training models...\n")

    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # LogisticRegression
    lrc.fit(X_train, y_train)

    # save best model
    model_path = os.path.join(MODELS_DIR, MODELS_NAME_DICT['random_forest'])
    joblib.dump(cv_rfc.best_estimator_, model_path)
    model_path = os.path.join(MODELS_DIR, MODELS_NAME_DICT['logistic'])
    joblib.dump(lrc, model_path)

    # Plot ROC curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    # plot logistic regression curve
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    # plot random forest curve
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test,
                   ax=ax, alpha=0.8)
    plt.savefig(os.path.join(RESULTS_DIR, RESULTS_NAME_DICT['roc_curve']),
                bbox_inches='tight')
    plt.close()

    # produces classification results report
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # produce and store feature importance plot
    output_path = os.path.join(RESULTS_DIR,
                               RESULTS_NAME_DICT['feature_importances'])
    feature_importance_plot(cv_rfc, X_test, output_path)

    print("\nTraining is completed. Models and results are saved.\n")


if __name__ == '__main__':
    # Import data
    bank_dataframe = import_data(DATA_PATH)

    # Perform EDA
    perform_eda(bank_dataframe)

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        dataframe=bank_dataframe, response='Churn')

    # Model training,prediction and evaluation
    train_models(X_train=X_TRAIN,
                 X_test=X_TEST,
                 y_train=Y_TRAIN,
                 y_test=Y_TEST)
