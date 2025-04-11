import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import logging
import traceback

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

# Log Streamlit app start
logging.info("UCLA Admission Predictor app started.")

# Set the page title and description
st.markdown("<h1 style='text-align: center;'>UCLA Admission Chances Predictor</h1>", unsafe_allow_html=True)
st.write("""
This app predicts whether a student is eligible for an admission to UCLA based on their grades and university rating.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()


# Load the pre-trained model
try:
    with open("models/MLPmodel.pkl", "rb") as mlp_pickle:
        mlp_model = pickle.load(mlp_pickle)
    logging.info("MLP model loaded successfully.")
except Exception as e:
    logging.error("Failed to load MLP model.")
    logging.error(traceback.format_exc())
    st.error("Error loading the prediction model. Please try again later.")
    st.stop()

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.markdown("<h3 style='text-align: center;'>Student Applicant Details</h3>", unsafe_allow_html=True)

    # Create 2 columns
    cols = st.columns(2)

    # Row 1
    # Column 1
    with cols[0]:
        GRE_Score = st.number_input("GRE Score", min_value=0, max_value=340, step=10)
    # Column 2
    with cols[1]:
        TOEFL_Score = st.number_input("TOEFL Score", min_value=0, max_value=120, step=10)

    # Row 2
    cols = st.columns(2)
    
    # Column 1
    with cols[0]:
        SOP = st.number_input("Statement of Purpose Strength", min_value=0.0, max_value=5.0, step=0.1)
    # Column 2
    with cols[1]:
        LOR = st.number_input("Letter of Recommendation Strength", min_value=0.0, max_value=5.0, step=0.1)

    # Row 3
    cols = st.columns(2)
    
    # Column 1
    with cols[0]:
        CGPA = st.number_input("CGPA", min_value=0.00, max_value=10.00, step=0.10)
    # Column 2
    with cols[1]:
        University_Rating = st.selectbox("University_Rating", options=["1", "2", "3", "4", "5"])

    # Row 4
    cols = st.columns(2)
    
    # Column 1
    with cols[0]:
        Research = st.selectbox("Reseach", options=["No", "Yes"])
    # Column 2
    with cols[1]:
        submitted = st.form_submit_button("Predict Admission Eligibility")

# Handle the dummy variables to pass to the model
if submitted:
    try:
        # Handle dependents
        University_Rating_1 = 1 if University_Rating == "1" else 0
        University_Rating_2 = 1 if University_Rating == "2" else 0
        University_Rating_3 = 1 if University_Rating == "3" else 0
        University_Rating_4 = 1 if University_Rating == "4" else 0
        University_Rating_5 = 1 if University_Rating == "5" else 0
        Research_0 = 1 if Research == "No" else 0
        Research_1 = 1 if Research == "Yes" else 0

        # Prepare the input for prediction. This has to go in the same order as it was trained
        prediction_input = [[GRE_Score, TOEFL_Score, SOP, LOR, CGPA,
            University_Rating_1, University_Rating_2, University_Rating_3,
            University_Rating_4, University_Rating_5, Research_0,
            Research_1
        ]]
        
        # Make prediction
        new_prediction = mlp_model.predict(prediction_input)

        # Display result
        st.subheader("Prediction Result:")
        if new_prediction[0] == 1:
            st.success("You are eligible for an admission at UCLA!")
        else:
            st.error("Sorry, you are not eligible for an admission at UCLA.")
        logging.info(f"Prediction completed: {new_prediction[0]}")

        try:
            st.image("loss_curve.png")
        except Exception as e:
            logging.warning("Could not display loss_curve.png image.")
            logging.warning(traceback.format_exc())

    except Exception as e:
        logging.error("Error occurred during prediction process.")
        logging.error(traceback.format_exc())
        st.error("An error occurred during prediction. Please check inputs and try again.")

st.write(
    """We used a machine learning (Multi Layer Perceptron) model to predict your admission eligibility."""
)
