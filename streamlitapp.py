import streamlit as st
from pymongo import MongoClient
import bcrypt
import base64


client = MongoClient("mongodb+srv://Apratim:wiINvlvnkfc5cRw4@atlascluster.mz70pny.mongodb.net/")
db = client["MaxBPM"]
users_collection = db["Users"]


def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def sign_up(username, password, email=None, age=None, sex=None, confirm_password=None):
    if users_collection.find_one({"username": username}):
        st.warning("Username already exists. Please choose a different one.")
    elif len(password) < 8:
        st.warning("Password must be at least 8 characters long.")
    elif password != confirm_password:
        st.warning("Passwords do not match.")
    else:
        hashed_password = hash_password(password)
        
        users_collection.insert_one({
            "username": username,
            "password": hashed_password,
            "email": email,
            "age": age,
            "sex": sex
        })
        st.success("Account created successfully! You can now log in.")
        st.session_state['logged_in'] = True
        st.session_state['username'] = username

# Function to update user profile
def update_profile(username, email, age, gender):
    users_collection.update_one(
        {"username": username},
        {"$set": {
            "email": email,
            "age": age,
            "gender": gender
        }}
    )
    st.success("Profile updated successfully.")


def get_user_info(username):
    user = users_collection.find_one({"username": username})
    return user

file_ = open("image[1].png", "rb").read()
base64_image = base64.b64encode(file_).decode()

rawHTML = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        width: 100%;
        height: 100%;
        background-image: url('data:image/jpeg;base64,{base64_image}');
        background-size: cover;
        background-position: center center;
    }}
    </style>
"""
st.markdown(rawHTML, unsafe_allow_html=True)

def authentication_form():
    st.title("Heart Disease Prediction App")
    
    with st.container():          
        auth_option = st.radio("Select Option", ["Log In", "Sign Up"])
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if auth_option == "Sign Up":
            email = st.text_input("Email (optional)")
            age = st.number_input("Age (optional)", min_value=0, max_value=120, step=1)
            sex = st.selectbox("Sex (optional)", options=["Select", "Male", "Female"])
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Sign Up"):
                sign_up(username, password, email, age, sex, confirm_password)
        else:
            if st.button("Log In"):
                log_in(username, password)
        
        st.markdown('</div>', unsafe_allow_html=True)  
# Function to log in a user
def log_in(username, password):
    user = users_collection.find_one({"username": username})
    if not user:
        st.error("Invalid username or password")
    elif bcrypt.checkpw(password.encode(), user['password']):
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.session_state['user_info'] = user  
    else:
        st.error("Invalid username or password")

# User profile page
def user_profile_page():
    st.title("User Profile")
    
    if 'username' in st.session_state:
        user_info = get_user_info(st.session_state['username'])
        
        if user_info:
           
            st.subheader("Personal Information")
            email = st.text_input("Email", value=user_info.get('email', ''))
            age = st.number_input("Age", min_value=0, max_value=120, step=1, value=user_info.get('age', 0))
            gender = st.selectbox("Gender", options=["Select", "Male", "Female", "Other"], index=["Select", "Male", "Female", "Other"].index(user_info.get('gender', "Select")))
            
            st.subheader("Heart Health Statistics")
            st.metric(label="Max Heart Rate", value=user_info.get('max_heart_rate', 'N/A'))
            avg_heart_rate = user_info.get('avg_heart_rate', 'N/A')
            min_heart_rate = user_info.get('min_heart_rate', 'N/A')
            st.metric(label="Average Heart Rate", value=avg_heart_rate)
            st.metric(label="Minimum Heart Rate", value=min_heart_rate)
            
            if st.button("Update Profile"):
                update_profile(st.session_state['username'], email, age, gender)
        else:
            st.error("User information could not be found.")
    else:
        st.error("User is not logged in.")

def main_app():    
    dashboard = st.Page("dashboard.py", title="My Heart Statistics")
    prediction = st.Page("heartDisease.py", title="Heart Disease Prediction")
    
    if st.sidebar.button("Log Out"):
        st.session_state.clear()
        st.session_state['logged_in'] = False  
        st.write("You have been logged out. Please log in again.")
        return 
    
    pg = st.navigation([dashboard, prediction])
    
    pg.run()

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    authentication_form()
else:
    main_app()
