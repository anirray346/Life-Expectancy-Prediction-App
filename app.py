import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(page_title="Life Expectancy Prediction", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts():
    with open('model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
imputation_values = artifacts['imputation_values']
unique_values = artifacts['unique_values']
encoded_columns = artifacts['encoded_columns']
skew_cols = artifacts['skew_cols']
outlier_bounds = artifacts['outlier_bounds']

# Title and Image
st.title("Life Expectancy Prediction App")
st.image("https://www.news-medical.net/image-handler/ts/20220301055452/ri/950/src/images/news/ImageForNews_705800_16461320916939377.jpg", width=500)

st.write("""
This app predicts the **Life Expectancy** of a population based on various health and economic factors.
Please adjust the values below to get a prediction.
""")

# Input Form
st.header("Enter Details")

# Organize inputs into columns
col1, col2, col3 = st.columns(3)

# Dictionary to hold user inputs
user_input = {}

# List of features expected by the logic (before encoding)
# We reconstruct the feature list from imputation_values keys (original columns)
feature_names = [f for f in imputation_values.keys() if f != 'Life expectancy']

for i, feature in enumerate(feature_names):
    # Determine column placement
    if i % 3 == 0:
        col = col1
    elif i % 3 == 1:
        col = col2
    else:
        col = col3
    
    with col:
        if feature in unique_values:
            # Categorical
            user_input[feature] = st.selectbox(feature, unique_values[feature])
        else:
            # Numerical
            # Use median as default value
            default_val = float(imputation_values[feature])
            user_input[feature] = st.number_input(feature, value=default_val)

# Predict Button
if st.button("Predict Life Expectancy"):
    # Create DataFrame from input
    input_df = pd.DataFrame([user_input])
    
    # 1. Impute (Although inputs are filled, good practice to ensure consistency)
    # in this app context, values come from user, but we treat them same way
    
    # 2. Encode
    input_df = pd.get_dummies(input_df, drop_first=True)
    # Ensure all columns from training exist
    for col in encoded_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    # Reorder columns to match training
    input_df = input_df[encoded_columns]
    
    # 3. Skewness Log Transform
    for col in skew_cols:
         if col in input_df.columns:
            # Only apply if value > 0 (log1p handles 0, but negative needs care if any)
            # Assuming widely positive values for these stats
            input_df[col] = np.log1p(input_df[col])
            
    # 4. Outlier Clipping
    for col, bounds in outlier_bounds.items():
        if col in input_df.columns:
            input_df[col] = np.clip(input_df[col], bounds['lower'], bounds['upper'])
            
    # 5. Scale
    input_scaled = scaler.transform(input_df)
    
    # 6. Predict
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"Predicted Life Expectancy: **{prediction:.2f} years**")

st.markdown("---")
st.header("Model Performance Summary")

# Performance Data
performance_data = {
    'Model': ['Random Forest', 'XGBoost', 'Linear Regression', 'Decision Tree'],
    'Test R2 Score': [0.968, 0.965, 0.964, 0.932],
    'RMSE': [1.67, 1.74, 1.77, 2.44]
}
perf_df = pd.DataFrame(performance_data)

st.write("Comparison of different models trained on the dataset:")
st.dataframe(perf_df)

st.subheader("Model Accuracy Comparison (R2 Score)")
st.bar_chart(perf_df.set_index('Model')['Test R2 Score'])
