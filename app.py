import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Model and Feature Mapping ---

# Load the saved pipeline
try:
    pipeline = joblib.load('elasticnet_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file 'elasticnet_pipeline.pkl' not found. Please ensure it's in the same directory.")
    st.stop()
    
# Load feature lookup and create mapping
# Load feature lookup and create mapping
try:
    lookup_df = pd.read_csv('feature_lookup.csv')
    
    # *** CORRECTION 1: Clean whitespace from the feature_code column ***
    lookup_df['feature_code'] = lookup_df['feature_code'].str.strip()
    
    DESCRIPTIVE_NAME_COLUMN = 'relevance' 
    
    # *** CORRECTION 2: Check for the descriptive column before setting index ***
    if DESCRIPTIVE_NAME_COLUMN not in lookup_df.columns:
        st.error(f"Feature lookup CSV is missing the required column '{DESCRIPTIVE_NAME_COLUMN}'. Actual columns are: {lookup_df.columns.tolist()}")
        st.stop()
        
    feature_map = lookup_df.set_index('feature_code')[DESCRIPTIVE_NAME_COLUMN].to_dict()
    
except FileNotFoundError:
    st.warning("Feature lookup file 'feature_lookup.csv' not found. Ensure it is in the same directory as app.py on GitHub.")
    feature_map = {}
except Exception as e:
    st.error(f"An unexpected error occurred during feature lookup: {e}")
    feature_map = {}


# --- 2. Define Features using the structure from your notebook ---

# These lists must match the columns expected by your model pipeline
numeric_features = [
    'f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f12', 
    'f13', 'f14', 'f15', 'f16', 'f18', 'f19', 'f20', 'f21', 'f22', 
    'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f30', 'f31', 'f32'
]
categorical_features = ['f2', 'f11', 'f17', 'f29', 'f33']


# --- 3. Streamlit UI Components ---
st.set_page_config(page_title="Relationship Probability Predictor", layout="centered")
st.title("💖 Relationship Probability Predictor")
st.markdown("Enter the feature values to predict the 'relationship_probability'.")

user_input = {}

# --- Numeric Features ---
st.header("Numeric Features")
cols_num = st.columns(3)
for i, feature_code in enumerate(numeric_features):
    # Get the descriptive name, default to code if not found
    display_name = feature_map.get(feature_code, feature_code)
    
    with cols_num[i % 3]:
        # The key in user_input MUST be the feature_code (e.g., 'f1'), not the display_name
        user_input[feature_code] = st.number_input(f'{display_name} ({feature_code})', value=0.0, step=0.01, format="%.4f")


# --- Categorical Features ---
st.header("Categorical Features")
cols_cat = st.columns(3)
for i, feature_code in enumerate(categorical_features):
    # Get the descriptive name
    display_name = feature_map.get(feature_code, feature_code)
    
    with cols_cat[i % 3]:
        # Use st.text_input for categorical features, converting to string as in the notebook
        user_input[feature_code] = st.text_input(f'{display_name} ({feature_code})', value="A") 


# --- 4. Prediction Button and Logic ---
if st.button("Predict Probability"):
    
    # 4a. Convert input to a DataFrame matching the structure the pipeline expects
    input_data = {
        col: [user_input[col]] for col in numeric_features + categorical_features
    }
    input_df = pd.DataFrame(input_data)
    
    # 4b. Ensure categorical features are of type string
    for col in categorical_features:
        input_df[col] = input_df[col].astype(str)

    try:
        # 4c. Make the prediction
        prediction = pipeline.predict(input_df)[0]
        
        # 4d. Clip the result to be between 0 and 100
        clipped_prediction = np.clip(prediction, 0, 100)
        
        st.success(f"### Predicted Relationship Probability: **{clipped_prediction:.2f}%**")
        
        # Optional: Display the raw prediction for comparison
        st.caption(f"Raw model output: {prediction:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check if the input values are correctly formatted.")

