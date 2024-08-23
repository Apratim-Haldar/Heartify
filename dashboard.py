import streamlit as st
from pymongo import MongoClient
import pandas as pd
import time
import base64

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
    </style>
"""

st.markdown(formBg, unsafe_allow_html=True)

def fetch_data():
    try:
        cluster = MongoClient("mongodb+srv://Apratim:wiINvlvnkfc5cRw4@atlascluster.mz70pny.mongodb.net/")
        db = cluster["MaxBPM"]
        collection = db["HeartRateMonitor"]
        data = list(collection.find())
        print(f"Fetched {len(data)} records from MongoDB")  
        return data
    except Exception as e:
        print(f"Error fetching data: {e}") 
        return []

st.title("Real-Time Heart Rate Monitor")

live_data_placeholder = st.empty()
placeholder = st.empty()

for seconds in range(200):
    data = fetch_data()
    df = pd.DataFrame(data)
   
    if not df.empty:
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Daily Resampling
        daily_trends = df.resample('D').agg({'maxBPM': 'max', 'minBPM': 'min', 'avgBPM': 'mean'})
        daily_trends.index = daily_trends.index.strftime('%Y-%m-%d')  

        # Weekly Resampling
        weekly_trends = df.resample('W').agg({'maxBPM': 'max', 'minBPM': 'min', 'avgBPM': 'mean'})
        weekly_trends.index = weekly_trends.index.strftime('%U, %B')  

        # Monthly Resampling
        monthly_trends = df.resample('ME').agg({'maxBPM': 'max', 'minBPM': 'min', 'avgBPM': 'mean'})
        monthly_trends.index = monthly_trends.index.strftime('%B %Y')  
        
        with live_data_placeholder.container():
            st.subheader('Live Heart Rate')
            st.line_chart(df[['maxBPM', 'avgBPM', 'minBPM']], use_container_width=True)
            time.sleep(1)

        with placeholder.container():
            col1,col2=st.columns(2)
            with col1:
                
                st.subheader('Daily Trends')
                st.line_chart(daily_trends[['maxBPM', 'minBPM', 'avgBPM']], use_container_width=True)
            with col2:
           
                st.subheader('Weekly Trends')
                st.line_chart(weekly_trends[['maxBPM', 'minBPM', 'avgBPM']], use_container_width=True)

            
            st.subheader('Monthly Trends')
            st.line_chart(monthly_trends[['maxBPM', 'minBPM', 'avgBPM']], use_container_width=True)
            time.sleep(1)
    else:
        print("No data available to display.")  
