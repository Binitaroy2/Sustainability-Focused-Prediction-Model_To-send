import os
import subprocess
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Project root for paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "updated_energy_dataset.csv")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # Load models directly with absolute paths
    scaler_ = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
    rf_ = joblib.load(os.path.join(MODELS_PATH, "best_rf_model.pkl"))
    cnn_ = load_model(os.path.join(MODELS_PATH, "cnn_model.keras"))
    rnn_ = load_model(os.path.join(MODELS_PATH, "rnn_model.keras"))
    return scaler_, rf_, cnn_, rnn_

def main():
    # Hide the GitHub link and toolbar in Streamlit Cloud
    # Hide the GitHub link and toolbar in Streamlit Cloud
    st.markdown("""
 <style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
  section[data-testid="stToolbar"] {
    display: none !important;
  }
  .reportview-container .main footer {visibility: hidden;}    
  .viewerBadge_container__1QSob {display: none !important;}
  [data-testid="collapsedControl"] {
    display: none !important;
  }
  .viewerBadge_link__1SdzA {
    display: none !important;
  }
  .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK {
   display: none;
 }
 </style>
 """, unsafe_allow_html=True)

    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability-Focused Energy Predictor")
    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            # Run train.py via subprocess to avoid import execution
            src_dir = os.path.join(PROJECT_ROOT, 'src')
            result = subprocess.run(['/home/adminuser/venv/bin/python', 'train.py'], capture_output=True, text=True, cwd=src_dir)
            if result.returncode == 0:
                st.success("✅ Model retrained! Click ‘Reload Models’ to pick up changes.")
            else:
                st.error(f"Training failed: {result.stderr}")
    if st.sidebar.button("Reload Models"):
        load_models.clear()  # Clear cache to force reload
        st.success("🔄 Models reloaded.")
    # Data preview
    df = load_data()
    st.markdown("### Raw data preview")
    st.dataframe(df.head())
    # Predictions on data
    scaler_, rf_, cnn_, rnn_ = load_models()
    selected_features = [
        'Energy_Production_MWh', 'Type_of_Renewable_Energy', 'Installed_Capacity_MW',
        'Energy_Storage_Capacity_MWh', 'Storage_Efficiency_Percentage', 'Grid_Integration_Level'
    ]  # 6 features
    numeric_features = [
        'Energy_Production_MWh',
        'Installed_Capacity_MW',
        'Energy_Storage_Capacity_MWh',
        'Storage_Efficiency_Percentage',
        'Grid_Integration_Level'
    ]  # 5 numeric
    try:
        X = df[selected_features]
        y = df["Energy_Consumption_MWh"]
        # Handle NaNs/infs if any
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        numeric_X = X[numeric_features]
        scaled_numeric = scaler_.transform(numeric_X)
        X_scaled = X.copy()
        X_scaled[numeric_features] = scaled_numeric
        Xs = X_scaled.values  # Full 6 features for model
        preds = rf_.predict(Xs)
        st.markdown("### RF: Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y, preds, alpha=0.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted (RF)")
        st.pyplot(fig)
    except ValueError as e:
        st.warning(f"Prediction error in bulk data: {e}. Click 'Retrain Model' to update models with current data features.")

    # Energy Consumption Predictor Form
    st.header("Energy Consumption Predictor")
    energy_production = st.number_input("Energy Production (MWh)")
    type_renewable = st.selectbox("Type of Renewable Energy", options=["Solar", "Wind", "Hydroelectric", "Biomass", "Geothermal", "Tidal", "Wave"])
    installed_capacity = st.number_input("Installed Capacity (MW)")
    energy_storage_capacity = st.number_input("Energy Storage Capacity (MWh)")
    storage_efficiency = st.number_input("Storage Efficiency (%)")
    grid_integration_level = st.number_input("Grid Integration Level")
    model_type = st.selectbox("Model Type:", options=["Random Forest", "CNN", "RNN"])

    if st.button("Predict"):
        try:
            # Map renewable type to numeric (based on typical encoding: 1=Solar, 2=Wind, etc.)
            type_map = {"Solar": 1, "Wind": 2, "Hydroelectric": 3, "Biomass": 4, "Geothermal": 5, "Tidal": 6, "Wave": 7}
            type_num = type_map.get(type_renewable, 1)  # Default to Solar if not found

            # Create input array matching training features (6 features)
            input_data = np.array([[energy_production, type_num, installed_capacity, energy_storage_capacity, storage_efficiency, grid_integration_level]])

            # Handle NaNs/infs
            input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale only numeric features (columns 0,2,3,4,5)
            numeric_input = input_data[:, [0,2,3,4,5]]
            scaled_numeric = scaler_.transform(numeric_input)
            input_scaled = input_data.copy()
            input_scaled[:, [0,2,3,4,5]] = scaled_numeric

            # Predict based on selected model
            if model_type == "Random Forest":
                prediction = rf_.predict(input_scaled)[0]
            elif model_type == "CNN":
                input_reshaped = input_scaled.reshape((1, input_scaled.shape[1], 1))
                prediction = cnn_.predict(input_reshaped)[0][0]
            elif model_type == "RNN":
                input_reshaped = input_scaled.reshape((1, input_scaled.shape[1], 1))
                prediction = rnn_.predict(input_reshaped)[0][0]

            st.success(f"Predicted Energy Consumption: {prediction:.2f} MWh")
        except ValueError as e:
            st.error(f"Prediction error: {e}. Ensure input values are valid numbers and match model expectations (6 features). Try retraining the model.")
        except Exception as e:
            st.error(f"Model error: {e}. The model architecture may not support the input shape. Try a different model or retrain.")

if __name__ == "__main__":
    main()
