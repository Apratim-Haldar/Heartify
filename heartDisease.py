import streamlit as st
import pickle
import numpy as np
from pymongo import MongoClient
import base64
import time

with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

def fetch_latest_maxHR():
    try:
        cluster = MongoClient("mongodb+srv://Apratim:wiINvlvnkfc5cRw4@atlascluster.mz70pny.mongodb.net/")
        db = cluster["MaxBPM"]
        collection = db["HeartRateMonitor"]
        latest_record = collection.find().sort("timestamp", -1).limit(1)
        for record in latest_record:
            return record.get("maxBPM", 150) 
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return 150 

latest_maxHR = fetch_latest_maxHR()

st.title("Heart Disease Prediction")
st.markdown("---")
st.markdown("### How does it work?")
st.markdown("This is Heartify's in built machine learning model that will help you predict likelihood of heart disease based on your current heart rate and a few other medical factors which you can get verified from your doctor.")

file_ = open("background.jpeg", "rb").read()
base64_image = base64.b64encode(file_).decode()

formBg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        width: 100%;
        height: 100%;
        background-image: url('data:image/jpeg;base64,{base64_image}');
        background-size: cover;
        background-position: center center;
        background-color: rgba(255, 255, 255, 0.8)
    }}
    [class="st-emotion-cache-1sno8jx e1nzilvr4"]{{
        
    }}
    [id="heart-disease-prediction"] {{
        font-size: 54px;  
        text-shadow: 4px 2px 4px #000000; 
        font-family: 'Tahoma', sans-serif;  
        background: rgba(255, 255, 255, 0.5);  /* Semi-transparent white */
        backdrop-filter: blur(2px);  /* Blur effect */
        border-radius: 15px;  /* Rounded corners */
        padding: 20px;  /* Padding inside the form */   
    }}
    
    [id="how-does-it-work"] {{
        font-size: 40px;  
        text-shadow: 2px 2px 4px #000000; 
        background: rgba(255, 255, 255, 0.5);  /* Semi-transparent white */
        backdrop-filter: blur(2px);  /* Blur effect */
        border-radius: 15px;  /* Rounded corners */
        padding: 20px;  /* Padding inside the form */
    }}
    
    [data-testid="stForm"] {{
        background: rgba(255, 255, 255, 0.5);  /* Semi-transparent white */
        backdrop-filter: blur(2px);  /* Blur effect */
        border-radius: 15px;  /* Rounded corners */
        padding: 20px;  /* Padding inside the form */
    }}
    
    [data-testid="stForm"]:hover {{
        background: rgba(255, 255, 255, 0.5);  /* Semi-transparent white */
        backdrop-filter: blur(2px);  /* Blur effect */
        border-radius: 15px;  /* Rounded corners */
        padding: 20px;  /* Padding inside the form */
    }}
    </style>
"""
st.markdown(formBg, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Change the color of all input labels to black */
    .stTextInput > label,
    .stNumberInput > label,
    .st
    .stTextArea > label,
    .stSelectbox > label,
    .stMultiSelect > label,
    .stRadio > label,
    .stMetric,
    .stCheckbox > label {
        color: black; 
        font-size: 20px;
        font-weight: bold; bold/* Set label color to black */
    }
    </style>
""", unsafe_allow_html=True)
form = st.empty()
with form.container():
    with st.form(key='user_input_form', clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=150, value=1)
        with col2:
            sex = st.selectbox("Sex", options=['M', 'F'])
        col1, col2 = st.columns(2)
        with col1:
            chest_pain = st.selectbox("Chest Pain Type", options=['ATA', 'NAP', 'ASY', 'TA'])
        with col2:
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
        col1, col2 = st.columns(2)
        with col1:
            cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
            
        liveHeartRate = st.empty()
       
        col1, col2 = st.columns(2)
        with col1:
            resting_ecg = st.selectbox("Resting ECG", options=['Normal', 'ST', 'LVH'])
        with col2:
            exercise_angina = st.selectbox("Exercise-Induced Angina", options=['Y', 'N'])
        col1, col2 = st.columns(2)
        with col1:
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-5.0, max_value=10.0, step=0.1, value=1.0)
        with col2:
            st_slope = st.selectbox("ST Slope", options=['Up', 'Flat', 'Down'])
        
        submit_button = st.form_submit_button("Predict")

        st.markdown('</div>', unsafe_allow_html=True)  

    if submit_button:
        if age == 0:
            st.error("Age cannot be zero")
        elif resting_bp == 0:
            st.error("Resting blood pressure cannot be zero")
        else:
            user_input = np.array([[sex, chest_pain, resting_ecg, exercise_angina, st_slope]])
            user_input_encoded = encoder.transform(user_input).flatten()
            
            features = np.array([age, resting_bp, cholesterol, fasting_bs, latest_maxHR, oldpeak])
            features = np.concatenate([features, user_input_encoded])
            
            prediction = model.predict([features])
            
            if prediction[0] == 1:
                st.error("Warning: This individual is likely to have heart disease.")
            else:
                st.success("This individual is not likely to have heart disease.")
while True:
    with liveHeartRate.container():
        st.metric(label="Current Heart Rate (Automatically fetched from your Heartify)", value=latest_maxHR)
        time.sleep(20)
