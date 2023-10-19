import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# data dumb store
import joblib
# to locate yaml file
import yaml
# to locate directore
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# utils
import util as utils

# 1. Load configuration file
params = utils.load_config()

# 2. Load Dataset
def load_dataset(params):
    "Function to Load Datasset"
    
    # load data
    x_train = joblib.load("data/raw/x_train.pkl")
    y_train = joblib.load("data/raw/y_train.pkl")
    x_valid = joblib.load("data/raw/x_valid.pkl")
    y_valid = joblib.load("data/raw/y_valid.pkl")
    x_test = joblib.load("data/raw/x_test.pkl")
    y_test = joblib.load("data/raw/y_test.pkl")

    # Concat data
    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    # Rename
    train_set.columns = params['map_rename_features']
    valid_set.columns = params['map_rename_features']
    test_set.columns = params['map_rename_features']

    return train_set, valid_set, test_set

# 3. One Hot Encoding
def encoding_cat_feature(data, fit=False, encoder=None):
    
    # get copy and reset_index
    data_copy = data.copy().reset_index(drop=True)
    target_col = data_copy['Failure_Type']
    data_copy = data_copy.drop('Failure_Type', axis=1)
    
    # category features
    cat_features = data_copy.select_dtypes(include='object').columns
    
    if fit:
        # Ohe initialization
        ohe = OneHotEncoder(handle_unknown='ignore', drop=None)
        
        # fit transform
        ohe.fit(data_copy[cat_features])
        encoder = ohe
        encoded_df = pd.DataFrame(ohe.transform(data_copy[cat_features]).toarray())
    else:
        # use existing encoder object to transform
        encoded_df = pd.DataFrame(encoder.transform(data_copy[cat_features]).toarray())

    # rename columns
    encoded_df.columns = encoder.get_feature_names_out(cat_features)
    
    # drop original cat feature
    dropped_data = data_copy.drop(cat_features, axis=1)
    
    #merge one-hot encoded columns back with original DataFrame
    final_df = dropped_data.join([encoded_df, target_col])
    
    return encoder, final_df





    return train_std


if __name__ == "__main__":
    print('Start Preprocessing Phase')

    # 1. Load configuration file
    params = utils.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(params)

    # 3.1 One Hot Encoding train set
    encoder, train_fin = encoding_cat_feature(data = train_set,
                                            fit = True)
    
    # 3.2 Encoding valid set
    _, valid_fin = encoding_cat_feature(data = valid_set,
                                        encoder = encoder)
    
    # 3.3 Encoding test set
    _, test_fin = encoding_cat_feature(data = test_set,
                                    encoder = encoder)
    
    # 4. Label Encoding
    le_failure_type = LabelEncoder()
    le_failure_type.fit(params["label_Failure Type"])

    # Transform label encoder
    train_fin['Failure_Type'] = le_failure_type.transform(train_fin['Failure_Type'])
    valid_fin['Failure_Type'] = le_failure_type.transform(valid_fin['Failure_Type'])
    test_fin['Failure_Type'] = le_failure_type.transform(test_fin['Failure_Type'])

    # Save data
    joblib.dump(train_fin.drop(columns = "Failure_Type"), "data/processed/x_train_feng.pkl")
    joblib.dump(train_fin["Failure_Type"], "data/processed/y_train_feng.pkl")
    joblib.dump(valid_fin.drop(columns = "Failure_Type"), "data/processed/x_valid_feng.pkl")
    joblib.dump(valid_fin["Failure_Type"], "data/processed/y_valid_feng.pkl")
    joblib.dump(test_fin.drop(columns = "Failure_Type"), "data/processed/x_test_feng.pkl")
    joblib.dump(test_fin["Failure_Type"], "data/processed/y_test_feng.pkl")
    joblib.dump(encoder, "model/encoder.pkl")
    joblib.dump(le_failure_type, "model/le_failure_type.pkl")

    print('Process End')


    



