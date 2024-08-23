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
# Function to fetch data from MongoDB
def fetch_data():
    try:
        cluster = MongoClient("mongodb+srv://Apratim:wiINvlvnkfc5cRw4@atlascluster.mz70pny.mongodb.net/")
        db = cluster["MaxBPM"]
        collection = db["HeartRateMonitor"]
        data = list(collection.find())
        print(f"Fetched {len(data)} records from MongoDB")  # Debugging line
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")  # Error handling
        return []

st.title("Real-Time Heart Rate Monitor")

# Create placeholders for the charts
live_data_placeholder = st.empty()
placeholder = st.empty()

for seconds in range(200):
    # Fetch data from MongoDB
    data = fetch_data()
    
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Ensure the DataFrame is not empty
    if not df.empty:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Daily Resampling
        daily_trends = df.resample('D').agg({'maxBPM': 'max', 'minBPM': 'min', 'avgBPM': 'mean'})
        daily_trends.index = daily_trends.index.strftime('%Y-%m-%d')  # Format for day-month-year

        # Weekly Resampling
        weekly_trends = df.resample('W').agg({'maxBPM': 'max', 'minBPM': 'min', 'avgBPM': 'mean'})
        weekly_trends.index = weekly_trends.index.strftime('%U, %B')  # Format for week number and month

        # Monthly Resampling
        monthly_trends = df.resample('ME').agg({'maxBPM': 'max', 'minBPM': 'min', 'avgBPM': 'mean'})
        monthly_trends.index = monthly_trends.index.strftime('%B %Y')  # Format for month and year

        # Clear the placeholder and render the trends charts
        
        with live_data_placeholder.container():
            st.subheader('Live Heart Rate')
            st.line_chart(df[['maxBPM', 'avgBPM', 'minBPM']], use_container_width=True)
            time.sleep(1)

        with placeholder.container():
            col1,col2=st.columns(2)
            with col1:
                # Daily Trends
                st.subheader('Daily Trends')
                st.line_chart(daily_trends[['maxBPM', 'minBPM', 'avgBPM']], use_container_width=True)
            with col2:
            # Weekly Trends
                st.subheader('Weekly Trends')
                st.line_chart(weekly_trends[['maxBPM', 'minBPM', 'avgBPM']], use_container_width=True)

            # Monthly Trends
            st.subheader('Monthly Trends')
            st.line_chart(monthly_trends[['maxBPM', 'minBPM', 'avgBPM']], use_container_width=True)
            time.sleep(1)
    else:
        print("No data available to display.")  # Debugging line