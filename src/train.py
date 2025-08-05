import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import joblib
import os
import tempfile
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM

# Redirect MLflow to temp dir to avoid permission issues
mlflow_tmp = os.path.join(tempfile.gettempdir(), "mlruns")
os.makedirs(mlflow_tmp, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlflow_tmp}")

# MLflow experiment
mlflow.set_experiment("energy_consumption_prediction")

df = pd.read_csv("../data/updated_energy_dataset.csv")
target = "Energy_Consumption_MWh"
selected_features = [
    'Energy_Production_MWh',
    'Type_of_Renewable_Energy',
    'Installed_Capacity_MW',
    'Energy_Storage_Capacity_MWh',
    'Storage_Efficiency_Percentage',
    'Grid_Integration_Level'
]

# Ensure mapping for categorical column
mapping = {
    '1': 'Solar',
    '2': 'Wind',
    '3': 'Hydroelectric',
    '4': 'Geothermal',
    '5': 'Biomass',
    '6': 'Tidal',
    '7': 'Wave'
}
# If using numerical encoding for model (which you are), keep the column as is.
# If you want one-hot encoding for DL, add preprocessing as needed.

X = df[selected_features]
y = df[target]

# Standardize all features except the categorical one
numeric_features = [f for f in selected_features if f != 'Type_of_Renewable_Energy']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

os.makedirs("../models", exist_ok=True)
joblib.dump(scaler, "../models/scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Random Forest
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mlflow.sklearn.log_model(rf, "rf_model")
    mlflow.log_metric("mse", mse)
    joblib.dump(rf, "../models/best_rf_model.pkl")
    print("RF MSE:", mse)

# Prepare for DL models
X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# 2. CNN Model
with mlflow.start_run(run_name="CNN"):
    cnn = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    cnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    cnn.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)
    cnn_mse = cnn.evaluate(X_test_reshaped, y_test, verbose=0)[0]
    mlflow.tensorflow.log_model(cnn, "cnn_model")
    mlflow.log_metric("mse", cnn_mse)
    cnn.save("../models/cnn_model.keras")
    print("CNN MSE:", cnn_mse)

# 3. RNN Model
with mlflow.start_run(run_name="RNN"):
    rnn = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(32, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    rnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    rnn.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)
    rnn_mse = rnn.evaluate(X_test_reshaped, y_test, verbose=0)[0]
    mlflow.tensorflow.log_model(rnn, "rnn_model")
    mlflow.log_metric("mse", rnn_mse)
    rnn.save("../models/rnn_model.keras")
    print("RNN MSE:", rnn_mse)
``````python
# src/streamlit_app.py
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
    numeric_features = [f for f in selected_features if f != 'Type_of_Renewable_Energy']  # 5 numeric
    try:
        X = df[selected_features]
        y = df["Energy_Consumption_MWh"]
        # Scale only numeric features
        numeric_X = X[numeric_features]
        # Handle NaNs/infs if any
        numeric_X = numeric_X.replace([np.inf, -np.inf], np.nan).fillna(0)
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

            # Scale only numeric features (columns 0,2,3,4,5)
            numeric_input = input_data[:, [0,2,3,4,5]]
            # Handle NaNs/infs
            numeric_input = np.nan_to_num(numeric_input, nan=0.0, posinf=0.0, neginf=0.0)
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