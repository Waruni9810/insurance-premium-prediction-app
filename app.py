import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from time import sleep

# Function to load or create the insurance model
def load_or_create_model(df):
    model_path = 'insurance_model.pkl'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
        X = df.drop('PremiumPrice', axis=1)
        y = df['PremiumPrice']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        
        # Display MSE only during training
        mse = np.mean((model.predict(X_test) - y_test) ** 2)
        st.info(f"Model trained with Mean Squared Error: {mse:.2f}")
        
    return model

# Function to simulate loading data with progress bar
def load_data_with_progress():
    st.markdown("### Loading data...")
    progress_bar = st.progress(0)
    for i in range(100):
        sleep(0.05)
        progress_bar.progress(i + 1)
    st.success("Data loaded successfully!")
    st.balloons()  # Display balloons animation
    sleep(1)  # Short pause to display the success message
    st.empty()  # Clear the success message after a short time
    

# Custom CSS for improved styling and button customization
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f0;
        font-family: Arial, sans-serif;
    }
    .main-header {
        color: #f06529;
        font-size: 3em;
        margin-bottom: 20px;
    }

    .main-header2 {
        color: #f06529;
        font-size: 3em;
        margin-bottom: 20px;
        text-align: center;
    }
    .sub-header {
        color: #f06529;
        font-size: 1.5em;
        margin-bottom: 30px;
    }

    .orange-boxsmall {
        background-color: #f06529;
        color: white;
        padding: 10px;
        border-radius: 30px;
        margin-top: auto;
        text-align: center;
        font-size: auto;
        max-width: 150px;
        margin-left: 1px;
        margin-right: auto;
    } 
    .grey-box {
        background-color: #7D7F7E;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 1.2em;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .grey-box-title {
        font-size: 1.25em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .header {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px 0;
    }
    .header img {
        height: 100px;
        margin-right: 20px;
    }
    .header h1 {
        font-size: 2.5em;
        color: #f06529;
        margin: 0;
    }
    .text-column {
        padding: 20px;
    }

    .stButton>button {
        background-color: #f06529;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 5px;
        cursor: pointer;
    }

    .stButton>button:hover {
        background-color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with logo
st.sidebar.image("logo.png", width=200)
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About Us', 'Prediction', 'Contact Us'])

if app_mode == 'Home':
    # Centered header with image and text
    st.markdown('<div class="header">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.image("home.png", width=330)
    
    with col2:
        st.markdown('<h1 class="main-header">Welcome to the Insurance Premium Prediction App!</h1>', unsafe_allow_html=True)
        st.write('<p style="text-align: left;">This app predicts your yearly insurance premium based on your health data.</p>', unsafe_allow_html=True)
        st.markdown('<div class="orange-boxsmall"><a href="#about-us" style="color: white; text-decoration: none;">Read more..</a></div>', unsafe_allow_html=True)

    # Merged grey box with title and description
    st.markdown(
        """
        <div class="grey-box">
            <div class="grey-box-title">Understanding Insurance Premium</div>
            <p style="text-align: justify;">
            Insurance premiums are influenced by various factors such as age, health conditions, and lifestyle choices. 
            This app helps you understand how these factors impact your insurance costs and provides personalized predictions based on your input.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    ) 
elif app_mode == 'About Us':
    st.markdown('<h1 id="about-us" class="main-header2">About Us</h1>', unsafe_allow_html=True)
    st.write(
        '<p style="text-align: justify;">'
        'PredictSure is a cutting-edge insurance technology company dedicated to providing users with precise, personalized insurance premium predictions. '
        'Our mission is to simplify the insurance process and make it more transparent for everyone. '
        'We leverage advanced machine learning algorithms to analyze health data and deliver accurate predictions tailored to each individual.'
        '</p>',
        unsafe_allow_html=True
    )
    st.image("about us.jpg", width=600, use_column_width=True)  # Add image after the About Us section

elif app_mode == 'Prediction':
    # Load data with progress bar
    load_data_with_progress()
    
    df = pd.read_csv("medical dataset/Medicalpremium.csv")
    model = load_or_create_model(df)
    
    st.markdown('<h1 id="prediction" class="main-header2">Insurance Premium Prediction</h1>', unsafe_allow_html=True)
    st.write(
        '<p style="text-align: justify;">'
        'Enter your details below to receive a personalized prediction of your yearly insurance premium. '
        'This tool uses a regression model to estimate your costs based on your health metrics.'
        '</p>',
        unsafe_allow_html=True
    )

    # Sidebar Inputs
    st.sidebar.header('Your Information')
    age = st.sidebar.number_input('Age:', min_value=18, max_value=100)
    diabetes = st.sidebar.radio('Diabetes:', ['No', 'Yes'])
    blood_pressure = st.sidebar.radio('Blood Pressure Problems:', ['No', 'Yes'])
    transplants = st.sidebar.radio('Any Transplants:', ['No', 'Yes'])
    chronic_diseases = st.sidebar.radio('Any Chronic Diseases:', ['No', 'Yes'])
    height = st.sidebar.number_input('Height (cm):', min_value=120, max_value=220)
    weight = st.sidebar.number_input('Weight (kg):', min_value=30, max_value=200)
    allergies = st.sidebar.radio('Known Allergies:', ['No', 'Yes'])
    cancer_history = st.sidebar.radio('History of Cancer in Family:', ['No', 'Yes'])
    surgeries = st.sidebar.number_input('Number of Major Surgeries:', min_value=0, max_value=10)

    # Convert Yes/No to 0/1
    diabetes = 1 if diabetes == "Yes" else 0
    blood_pressure = 1 if blood_pressure == "Yes" else 0
    transplants = 1 if transplants == "Yes" else 0
    chronic_diseases = 1 if chronic_diseases == "Yes" else 0
    allergies = 1 if allergies == "Yes" else 0
    cancer_history = 1 if cancer_history == "Yes" else 0

    # Calculate BMI
    bmi = weight / (height / 100) ** 2

    # Prediction button
    if st.sidebar.button('Predict Cost'):
        
        with st.spinner('Predicting...'):
            user_data = pd.DataFrame({
                'Age': [age],
                'Diabetes': [diabetes],
                'BloodPressureProblems': [blood_pressure],
                'AnyTransplants': [transplants],
                'AnyChronicDiseases': [chronic_diseases],
                'Height': [height],
                'Weight': [weight],
                'KnownAllergies': [allergies],
                'HistoryOfCancerInFamily': [cancer_history],
                'NumberOfMajorSurgeries': [surgeries],
                'BMI': [bmi]
            })

            prediction = model.predict(user_data)
            prediction_lkr = prediction[0] * 4.35  # Convert to LKR
            
            st.markdown(f'<h3 style="color: #007BFF;">Predicted Yearly Premium Price: LKR {prediction_lkr:.2f}</h3>', unsafe_allow_html=True)


            # Display Graphs after Prediction
            fig, ax = plt.subplots()
            ax.bar(['Predicted Price'], [prediction_lkr], color="skyblue")
            ax.set_ylabel('Premium Price (LKR)')
            ax.set_title('Predicted Insurance Premium')
            st.pyplot(fig)

elif app_mode == 'Contact Us':
    st.markdown('<h1 id="contact-us" class="main-header2">Contact Us</h1>', unsafe_allow_html=True)
    st.write(
        '<p style="text-align: justify;">'
        'For any inquiries, support, or feedback, feel free to reach out to us at PredictSure. '
        'You can contact us via email at support@predictsure.com or call us at +94-123-456-789. '
        'Our team is always ready to assist you with your insurance-related questions and concerns.'
        '</p>',
        unsafe_allow_html=True
    ) 
    st.image("cover-new.png", width=600, use_column_width=True)  # Add image after the Contact Us section