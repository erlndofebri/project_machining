import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split

# data dumb store
import joblib

# to locate yaml file
import yaml

# to locate directore
import os

# import util
import util as utils

# 1. Load configuration file
params = utils.load_config()


# 2. read data
def read_data(path):
    "Function to read data with csv formar"
    data = pd.read_csv(path)
    return data

# 3. Handling Failure Type
def handling_failure_type(data, replacement):
    "Function to convert target features"

    data['Failure Type'] = data['Failure Type'].replace(replacement)
    
    return data

# 4. data defence
def check_data(input_data, params):
    "Function to Defense the Data"
    # check data types
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."
    assert input_data.select_dtypes("int").columns.to_list() == params["int64_columns"], "an error occurs in int32 column(s)."
    assert input_data.select_dtypes("float").columns.to_list() == params["float64_columns"], "an error occurs in int32 column(s)."

    # check range of data
    assert set(input_data['Type']).issubset(set(params["range_Type"])), "an error occurs in Type range."
    assert input_data['Rotational speed [rpm]'].between(params["range_Rotational speed [rpm]"][0], params["range_Rotational speed [rpm]"][1]).sum() == len(input_data), "an error occurs in Rotational speed [rpm] range."
    assert input_data['Tool wear [min]'].between(params["range_Tool wear [min]"][0], params["range_Tool wear [min]"][1]).sum() == len(input_data), "an error occurs in Tool wear [min] range."
    assert input_data['Air temperature [K]'].between(params["range_Air temperature [K]"][0], params["range_Air temperature [K]"][1]).sum() == len(input_data), "an error occurs in Air temperature [K] range."
    assert input_data['Process temperature [K]'].between(params["range_Process temperature [K]"][0], params["range_Process temperature [K]"][1]).sum() == len(input_data), "an error occurs in Process temperature [K] range."
    assert input_data['Torque [Nm]'].between(params["range_Torque [Nm]"][0], params["range_Torque [Nm]"][1]).sum() == len(input_data), "an error occurs in Torque [Nm] range."

# 5. Data Spliting
def split_data(input_data, config):
    "Function to split data into train, valid, and test"
    
    # Split predictor and label
    x = input_data[config["predictors"]].copy()
    y = input_data[config["label"]].copy()

    # 1st split train and test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 42,
        stratify = y
    )

    # 2nd split test and valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = config["valid_size"],
        random_state = 42,
        stratify = y_test
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    print("Start initiate Data Pipeline Process")

    # 1. Load configuration file
    params = utils.load_config()

    # 2. read data
    df = read_data(path = params['dataset_path'])

    # 3. Handling Failure Type
    handling_failure_type(data = df, 
                          replacement = params['mapping_target_feature'])
    
    # 4. data defense
    check_data(input_data = df, 
               params = params)
    
    # 5. Split the data
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = split_data(input_data = df, 
                                              config = params)
    
    # 6. Save data
    joblib.dump(x_train, "data/raw/x_train.pkl")
    joblib.dump(y_train, "data/raw/y_train.pkl")
    joblib.dump(x_valid, "data/raw/x_valid.pkl")
    joblib.dump(y_valid, "data/raw/y_valid.pkl")
    joblib.dump(x_test, "data/raw/x_test.pkl")
    joblib.dump(y_test, "data/raw/y_test.pkl")

    



