import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved pipeline
# The pipeline includes preprocessing (Yeo-Johnson, Scaling, OneHotEncoding) and the ElasticNet model
try:
    pipeline = joblib.load('elasticnet_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file 'elasticnet_pipeline.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Get the feature names from your training data (X)
# In your notebook, X contains the feature data. You'll need to define numeric and categorical features again.
# Assuming you have access to the original feature lists:
numeric_features = ['f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f12', 'f13', 'f14', 'f15', 'f16', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f30', 'f31', 'f32']
categorical_features = ['f2', 'f11', 'f17', 'f29', 'f33']

# --- Streamlit UI Components ---
st.set_page_config(page_title="Relationship Probability Predictor", layout="centered")
st.title("💖 Relationship Probability Predictor")
st.markdown("Enter the feature values to predict the 'relationship_probability'.")

# Dictionary to hold user inputs
user_input = {}

# Layout the input fields using columns for a cleaner look
st.header("Numeric Features")
cols_num = st.columns(3)
for i, feature in enumerate(numeric_features):
    with cols_num[i % 3]:
        # Use st.number_input for numeric features
        # Min/max values are placeholders and should be adjusted based on your data distribution
        user_input[feature] = st.number_input(f'{feature}', value=0.0, step=0.01)

st.header("Categorical Features")
cols_cat = st.columns(3)
# For categorical features, you'll need the unique values (levels) from your training data.
# Since you're using OneHotEncoder(handle_unknown="ignore"), a single text input is safer
# but a selectbox with known options is better for a production app.
# For simplicity and to match the model, let's use text input and assume user knows the possible values.
for i, feature in enumerate(categorical_features):
    with cols_cat[i % 3]:
        # Use st.text_input for categorical features, converting to string as in the notebook
        user_input[feature] = st.text_input(f'{feature}', value="A") 

# --- Prediction Button and Logic ---
if st.button("Predict Probability"):
    # Convert input to a DataFrame matching the structure the pipeline expects
    input_df = pd.DataFrame([user_input])
    
    # Ensure categorical features are of type string, matching the notebook preprocessing
    for col in categorical_features:
        input_df[col] = input_df[col].astype(str)

    try:
        # Make the prediction
        prediction = pipeline.predict(input_df)[0]
        
        # Clip the result to be between 0 and 100, as done in the notebook
        clipped_prediction = np.clip(prediction, 0, 100)
        
        st.success(f"### Predicted Relationship Probability: **{clipped_prediction:.2f}%**")
        
        # Optional: Display the raw prediction for comparison
        st.caption(f"Raw model output: {prediction:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check if the categorical feature inputs are valid values that the model has seen.")