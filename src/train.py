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

# Use temp dir for model saving to avoid permission issues
models_dir = os.path.join(tempfile.gettempdir(), "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Random Forest
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mlflow.sklearn.log_model(rf, "rf_model")
    mlflow.log_metric("mse", mse)
    joblib.dump(rf, os.path.join(models_dir, "best_rf_model.pkl"))
    print("RF MSE:", mse)

# Prepare for DL models
X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# 2. CNN Model
with mlflow.start_run(run_name="CNN"):
    cnn = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
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
    cnn.save(os.path.join(models_dir, "cnn_model.keras"))
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
    rnn.save(os.path.join(models_dir, "rnn_model.keras"))
    print("RNN MSE:", rnn_mse)