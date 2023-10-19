from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.dummy import DummyClassifier


import numpy as np

# utils
import util as utils

from datetime import datetime
from tqdm import tqdm
import yaml
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import hashlib


# 1. Load configuration file
params = utils.load_config()

# 2. Load Dataset
def load_dataset(params):
    "Function to Load Datasset"

    # Load Data
    x_train = joblib.load("data/processed/x_train_feng.pkl")
    y_train = joblib.load("data/processed/y_train_feng.pkl")

    x_valid = joblib.load("data/processed/x_valid_feng.pkl")
    y_valid = joblib.load("data/processed/y_valid_feng.pkl")

    x_test = joblib.load("data/processed/x_test_feng.pkl")
    y_test = joblib.load("data/processed/y_test_feng.pkl")

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# 3. Create Model Param
def create_model_param():
    """Create the model objects"""
    knn_params = {
        'n_neighbors': [50, 100, 200],
    }
    
    lgr_params = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1],
        'max_iter': [100, 300, 500]
    }

    dt_params = {
        'min_samples_split': [2, 5, 10, 25, 50]
    }

    # Create model params
    list_of_param = {
        'KNeighborsClassifier': knn_params,
        'LogisticRegression': lgr_params,
        'DecisionTreeClassifier': dt_params
    }

    return list_of_param

# 4. Create Model Object
def create_model_object():
    """Create the model objects"""
    print("Creating model objects")

    # Create model objects
    knn = KNeighborsClassifier()
    lgr = LogisticRegression(solver='liblinear')
    dt = DecisionTreeClassifier()

    # Create list of model
    list_of_model = [
        {'model_name': knn.__class__.__name__, 'model_object': knn},
        {'model_name': lgr.__class__.__name__, 'model_object': lgr},
        {'model_name': dt.__class__.__name__, 'model_object': dt}
    ]

    return list_of_model


# 5. Training Model
def train_model(return_file=True):
    """Function to get the best model"""
    # Load dataset
    X_train = joblib.load(params['train_feng_set_path'][0])
    y_train = joblib.load(params['train_feng_set_path'][1])
    X_valid = joblib.load(params['valid_feng_set_path'][0])
    y_valid = joblib.load(params['valid_feng_set_path'][1])
    
    # Create list of params & models
    list_of_param = create_model_param()
    list_of_model = create_model_object()

    # List of trained model
    list_of_tuned_model = {}

    # Train model
    for base_model in list_of_model:
        # Current condition
        model_name = base_model['model_name']
        model_obj = copy.deepcopy(base_model['model_object'])
        model_param = list_of_param[model_name]

        # Debug message
        print('Training model :', model_name)

        # Create model object
        model = RandomizedSearchCV(estimator = model_obj,
                                   param_distributions = model_param,
                                   n_iter=5,
                                   cv = 5,
                                   random_state = 123,
                                   n_jobs=1,
                                   verbose=10,
                                   scoring = 'roc_auc')
        
        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_valid = model.predict_proba(X_valid)[:, 1]
        
        # Get score
        train_score = roc_auc_score(y_train, y_pred_proba_train)
        valid_score = roc_auc_score(y_valid, y_pred_proba_valid)

        # Append
        list_of_tuned_model[model_name] = {
            'model': model,
            'train_auc': train_score,
            'valid_auc': valid_score,
            'best_params': model.best_params_
        }

        print("Done training")
        print("")
   
    # Dump data
    joblib.dump(list_of_param, params['list_of_param_path'])
    joblib.dump(list_of_model, params['list_of_model_path'])
    joblib.dump(list_of_tuned_model, params['list_of_tuned_model_path'])


    if return_file:
        return list_of_param, list_of_model, list_of_tuned_model 

# 6. Get Best Model
def get_best_model(return_file=True):
    """Function to get the best model"""
    # Load tuned model
    list_of_tuned_model = joblib.load(params['list_of_tuned_model_path'])

    # Get the best model
    best_model_name = None
    best_model = None
    best_performance = -99999
    best_model_param = None

    for model_name, model in list_of_tuned_model.items():
        if model['valid_auc'] > best_performance:
            best_model_name = model_name
            best_model = model['model']
            best_performance = model['valid_auc']
            best_model_param = model['best_params']

    # Dump the best model
    joblib.dump(best_model, params['best_model_path'])

    # Print
    print('=============================================')
    print('Best model        :', best_model_name)
    print('Metric score      :', best_performance)
    print('Best model params :', best_model_param)
    print('=============================================')

    if return_file:
        return best_model
    

# 7. Train Model
def train_model(return_file=True):
    """Function to get the best model"""
    # Load dataset
    X_train = joblib.load(params['train_feng_set_path'][0])
    y_train = joblib.load(params['train_feng_set_path'][1])
    X_valid = joblib.load(params['valid_feng_set_path'][0])
    y_valid = joblib.load(params['valid_feng_set_path'][1])
    
    # Create list of params & models
    list_of_param = create_model_param()
    list_of_model = create_model_object()

    # List of trained model
    list_of_tuned_model = {}

    # Train model
    for base_model in list_of_model:
        # Current condition
        model_name = base_model['model_name']
        model_obj = copy.deepcopy(base_model['model_object'])
        model_param = list_of_param[model_name]

        # Debug message
        print('Training model :', model_name)

        # Create model object
        model = RandomizedSearchCV(estimator = model_obj,
                                   param_distributions = model_param,
                                   n_iter=5,
                                   cv = 5,
                                   random_state = 123,
                                   n_jobs=1,
                                   verbose=10,
                                   scoring = 'roc_auc')
        
        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_valid = model.predict_proba(X_valid)[:, 1]
        
        # Get score
        train_score = roc_auc_score(y_train, y_pred_proba_train)
        valid_score = roc_auc_score(y_valid, y_pred_proba_valid)

        # Append
        list_of_tuned_model[model_name] = {
            'model': model,
            'train_auc': train_score,
            'valid_auc': valid_score,
            'best_params': model.best_params_
        }

        print("Done training")
        print("")
   
    # Dump data
    joblib.dump(list_of_param, params['list_of_param_path'])
    joblib.dump(list_of_model, params['list_of_model_path'])
    joblib.dump(list_of_tuned_model, params['list_of_tuned_model_path'])


    if return_file:
        return list_of_param, list_of_model, list_of_tuned_model 

# 8. Get Best Threshold
def get_best_threshold(return_file=True):
    """Function to tune & get the best decision threshold"""
    # Load data & model
    x_valid = joblib.load(params['valid_feng_set_path'][0])
    y_valid = joblib.load(params['valid_feng_set_path'][1])
    best_model = joblib.load(params['best_model_path'])

    # Get the proba pred
    y_pred_proba = best_model.predict_proba(x_valid)[:, 1]

    # Initialize
    metric_threshold = pd.Series([])
    
    # Optimize
    for threshold_value in THRESHOLD:
        # Get predictions
        y_pred = (y_pred_proba >= threshold_value).astype(int)

        # Get the F1 score
        metric_score = f1_score(y_valid, y_pred, average='macro')

        # Add to the storage
        metric_threshold[metric_score] = threshold_value

    # Find the threshold @max metric score
    metric_score_max_index = metric_threshold.index.max()
    best_threshold = metric_threshold[metric_score_max_index]
    print('=============================================')
    print('Best threshold :', best_threshold)
    print('Metric score   :', metric_score_max_index)
    print('=============================================')
    
    # Dump file
    joblib.dump(best_threshold, params['best_threshold_path'])

    if return_file:
        return best_threshold




if __name__ == "__main__":
    print('Start Preprocessing Phase')

    # 1. Load configuration file
    params = utils.load_config()

    # 2. Load dataset
    train_set, y_train, valid_set, y_valid, test_set, y_test = load_dataset(params)

    # 3. Training
    list_of_param, list_of_model, list_of_tuned_model = train_model()
    
    # 4. Get Best Model
    best_model = get_best_model()

    # 5. Get Best Threshold
    THRESHOLD = np.linspace(0, 1, 100)
    get_best_threshold()

    print('Process End')
    