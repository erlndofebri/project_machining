from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import util as utils
import data_pipeline as data_pipeline
import preprocessing as preprocessing
import yaml
import joblib


params = utils.load_config()
model_data = joblib.load(params["production_model_path"])
le_encoder = joblib.load(params["le_encoder_path"])
ohe_encoder = joblib.load(params["ohe_stasiun_path"])

class api_data(BaseModel):
    Air_temperature : float 
    Process_temperature : float
    Rotational_speed : float
    Torque : float
    Tool_wear : int
    Type_H : int
    Type_M : int
    Type_L : int 

# FASTAPI
app = FastAPI()

@app.get("/")
def home():
    return "FastAPI is up!"

@app.post("/predict/")
def get_data(data: api_data):
    try:
        input_list = [
            data.Air_temperature, data.Process_temperature, 
            data.Rotational_speed, data.Torque, data.Tool_wear, 
            data.Type_H, data.Type_L, data.Type_M
        ]

        # Reshape input data to match the shape used during training
        input_data = [input_list]

        result_prediction = model_data.predict(input_data)[0]

        if result_prediction == 0:
            # Jika hasil prediksi adalah 0
            print("Hasil customer adalah Failure")
            return {'prediction_is': 'Failure',
                    'error_msg': ' '}
        else:
            # Jika hasil prediksi adalah 1
            print("Hasil customer adalah No Failure")
            return {'prediction_is': 'No Failure', 
                    'error_msg': ' '}

    except ZeroDivisionError:
    # Tangani ZeroDivisionError jika terjadi
        return {'prediction_is':' ',
                 "error_msg": "Cannot divide by zero"}
    
# def predict(data: api_data):
#     # convert data api to dataframe
#     data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

#     # check range data
#     try:
#         data_pipeline.check_data(data, params)
#     except AssertionError as ae:
#         return {"res": [], "error_msg": str(ae)}
    


#     # one hot encoding
#     # data = preprocessing.encoding_cat_feature(data = data,
#     #                                           encoder = ohe_encoder)
#     # predict data
#     y_pred = model_data.predict(data)

#     # inverse transform
#     y_pred = list(le_encoder.inverse_transform(y_pred))[0]

#     return {"res":y_pred, "error_msg": ""}




if __name__ == "__main__":
    uvicorn.run("api:app", 
                host = "192.168.1.147", 
                port = 8080)