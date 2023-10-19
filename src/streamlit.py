import streamlit as st
import requests
import joblib

# Add some information about the service
st.title("Machine Failure Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# create form of input
with st.form(key = "air_data_form"):
    # create box number of input
    Air_temperature = st.number_input(
        label = "1.\tEnter Air temperature Value [K]:",
        min_value = 295.3,
        max_value = 304.5,
        help = "Value range from 295.3 to 304.5"
    )

    Process_temperature = st.number_input(
        label = "2.\tEnter Process temperature [K]:",
        min_value = 305.7,
        max_value = 313.8,
        help = "Value range from 305.7 to 313.8"
    )

    Rotational_speed = st.number_input(
        label = "3.\tEnter Rotational speed [rpm]:",
        min_value = 1168.0,
        max_value = 2886.0,
        help = "Value range from 1168.0 to 2886.0"
    )

    Torque = st.number_input(
        label ="4.\tEnter Torque [Nm]:",
        min_value = 3.8,
        max_value = 76.6,
        help = "Value range between 3.8 to 76.6"
    )

    Tool_wear = st.number_input(
        label = "5.\tEnter Tool wear [min]:",
        min_value = 0,
        max_value = 253,
        help = "Value range between 0 to 253"
    )

    Type_H = st.number_input(
        label = "6. Enter Type_H:",
        min_value = 0,
        max_value = 1,
        help = "Value range between 0 to 1"
    )

    Type_L = st.number_input(
        label="7. Enter Type_L:",
        min_value=0,
        max_value=1,
        help="Value range between 0 to 1",
        key="Type_L"  
    )

    Type_M = st.number_input(
        label="8. Enter Type_M:",
        min_value=0,
        max_value=1,
        help="Value range between 0 to 1",
        key="Type_M"  
)


    
    # Create submit buttin to the form
    submitted = st.form_submit_button("Predict")

    # Condition when form was submitted
    if submitted:
        # create dict of all data in the form
        raw_data = {
            "Air_temperature" : Air_temperature,
            "Process_temperature" : Process_temperature,
            "Rotational_speed" : Rotational_speed,
            "Torque" : Torque,
            "Tool_wear" : Tool_wear,
            "Type_H" : Type_H,
            "Type_L" : Type_L,
            "Type_M" : Type_M
        }

        # Create loading animation while predicting
        with st.spinner("Predicting on process ..."):
            res = requests.post("http://api_backend:8080/predict", 
                                json=raw_data).json()
            print(res)
            y_pred = res['prediction_is']
            st.success(y_pred)

            # # Parse the prediction result
            # if res["error_msg"] != "":
            #     st.error("Error Occurs While Predicting: {}").format(res['error_msg'])
            # else:
            #     if res['res'] != "yes":
            #         st.warning("Predicted will not failure")
            #     else:
            #         st.success("Predicted will failure")