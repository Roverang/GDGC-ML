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
try:
    lookup_df = pd.read_csv('feature_lookup.csv')
    
    # Map feature_code column to uppercase and clean for dictionary key consistency
    lookup_df['feature_code'] = lookup_df['feature_code'].str.strip().str.upper()
    
    # Map feature_code to relevance (the real descriptive name)
    feature_map = lookup_df.set_index('feature_code')['relevance'].to_dict()
except FileNotFoundError:
    st.warning("Feature lookup file 'feature_lookup.csv' not found. Using generic feature codes.")
    feature_map = {}
except KeyError:
    # Using 'relevance' as the descriptive column name
    st.error("Feature lookup CSV has incorrect column names. Expected 'feature_code' and 'relevance'.")
    st.stop()


# --- 2. Define Features using the structure from your notebook (NOW ALL UPPERCASE) ---

numeric_features = [
    'F1', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F12', 
    'F13', 'F14', 'F15', 'F16', 'F18', 'F19', 'F20', 'F21', 'F22', 
    'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F30', 'F31', 'F32'
]
categorical_features = ['F2', 'F11', 'F17', 'F29', 'F33']


# --- 3. Streamlit UI Components ---
st.set_page_config(page_title="Relationship Probability Predictor", layout="centered")
st.title("💖 Relationship Probability Predictor")
st.markdown("Enter the feature values to predict the 'relationship_probability'.")

user_input = {}

# --- Numeric Features ---
st.header("Numeric Features")
cols_num = st.columns(3)
for i, feature_code in enumerate(numeric_features):
    # Get the descriptive name, default to code if not found (Keys are now uppercase)
    display_name = feature_map.get(feature_code, feature_code)
    
    with cols_num[i % 3]:
        # UI uses uppercase feature_code
        user_input[feature_code] = st.number_input(f'{display_name} ({feature_code})', value=0.0, step=0.01, format="%.4f")


# --- Categorical Features ---
st.header("Categorical Features")
cols_cat = st.columns(3)
for i, feature_code in enumerate(categorical_features):
    # Get the descriptive name (Keys are now uppercase)
    display_name = feature_map.get(feature_code, feature_code)
    
    with cols_cat[i % 3]:
        # UI uses uppercase feature_code
        user_input[feature_code] = st.text_input(f'{display_name} ({feature_code})', value="A") 


# --- 4. Prediction Button and Logic ---
if st.button("Predict Probability"):
    
    # 4a. Convert input to a DataFrame using the UPPERCASE keys from user_input
    input_data = {
        col: [user_input[col]] for col in numeric_features + categorical_features
    }
    input_df = pd.DataFrame(input_data)
    
    # 🚨 CRITICAL FIX: Convert all column names to LOWERCASE to match the trained model's expectation.
    input_df.columns = input_df.columns.str.lower()
    
    # 4b. Ensure categorical features are of type string
    # We use the lowercase names here (f2, f11, etc.)
    lowercase_cat_features = [f.lower() for f in categorical_features]
    for col in lowercase_cat_features:
        input_df[col] = input_df[col].astype(str)

    try:
        # 4c. Make the prediction (pipeline expects lowercase columns)
        prediction = pipeline.predict(input_df)[0]
        
        # 4d. Clip the result to be between 0 and 100
        clipped_prediction = np.clip(prediction, 0, 100)
        
        st.success(f"### Predicted Relationship Probability: **{clipped_prediction:.2f}%**")
        
        # Optional: Display the raw prediction for comparison
        st.caption(f"Raw model output: {prediction:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check if the input values are correctly formatted.")
