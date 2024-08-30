# Insurance Premium Prediction App

## Overview
The Insurance Premium Prediction App is a web-based tool built with Streamlit that predicts yearly insurance premiums based on user-provided health data. The app uses a machine learning model (Linear Regression) to estimate the insurance premium in Sri Lankan Rupees (LKR). It also provides a graphical visualization of the predicted premium.

## Features
- **Home Page**: Introduction to the app and a brief overview of how it works.
- **About Us**: Information about the organization behind the app.
- **Prediction**: Users can input their health data to get a personalized insurance premium prediction.
- **Contact Us**: Contact information for further inquiries or support.

## How to Run the App
1. **Install dependencies**: Ensure you have Python installed and use the following command to install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the app**: Navigate to the directory containing the app script and run:

    ```bash
    streamlit run app.py
    ```

3. **Open the app**: The app will automatically open in your default web browser. If not, go to `http://localhost:8501` in your browser.

## Files
- **app.py**: The main Python script that runs the Streamlit app.
- **requirements.txt**: Lists all the Python dependencies needed to run the app.
- **README.md**: Documentation about the app, including setup instructions and features.

## App Workflow
1. **Model Creation**:
   - The app loads an existing Linear Regression model from `insurance_model.pkl`.
   - If the model does not exist, it trains a new model using the provided dataset (`Medicalpremium.csv`), calculates the Mean Squared Error (MSE), and saves the model.
   
2. **Prediction**:
   - Users enter their health data into the provided fields in the sidebar.
   - When users click "Predict Cost," the app calculates the BMI and uses the model to predict the yearly insurance premium.
   - The predicted premium is displayed along with a graphical representation.

3. **Visualization**:
   - After the prediction, the app generates a bar graph showing the predicted insurance premium in LKR.

## Screenshots

![Home](/screenshot/Home.png)
![Prediction](/screenshot/prediction.png)
![About Us](/screenshot/About%20Us.png)
![Contact Us](/screenshot/contact%20us.png)


## Kaggle Dataset

Download it (https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction/data).

## Deployment

The application is deployed on Streamlit Cloud. Access it (https://insurance-premium-prediction-app-fiwtksjejujjz7gtqgadtr.streamlit.app/).

## Repository

GitHub Repository: [Medical Insurance Prediction](https://github.com/Waruni9810/insurance-premium-prediction-app)

