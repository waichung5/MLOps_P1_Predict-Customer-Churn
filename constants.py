# full path of data file
DATA_PATH = "./data/bank_data.csv"

# directory for storing EDA images
EDA_IMAGES_DIR = "./images/eda"

# dictionary of EDA images' names
EDA_NAME_DICT = {
    'Churn': 'churn_distribution.png',
    'Customer_Age': 'customer_age_distribution.png',
    'Marital_Status': 'marital_status_distribution.png',
    'Total_Trans_Ct': 'total_transaction_distribution.png',
    'Heatmap': 'heatmap.png'
}

# categories for encoding
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

# categories for feature engineering
KEEP_COLS = ['Customer_Age', 'Dependent_count',
             'Months_on_book', 'Total_Relationship_Count',
             'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
             'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
             'Total_Trans_Amt', 'Total_Trans_Ct',
             'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn',
             'Marital_Status_Churn', 'Income_Category_Churn',
             'Card_Category_Churn']

# directory for storing trained models
MODELS_DIR = "./models"

# dictionary of models' names
MODELS_NAME_DICT = {
    'random_forest': 'rfc_model.pkl',
    'logistic': 'logistic_model.pkl'
}

# directory for storing training evaluation results
RESULTS_DIR = "./images/results"

# dictionary of result files' names
RESULTS_NAME_DICT = {
    'roc_curve': 'roc_curve_result.png',
    'random_forest': 'rf_results.png',
    'logistic': 'logistic logistic_results.png',
    'feature_importances': 'feature_importances.png'
}
