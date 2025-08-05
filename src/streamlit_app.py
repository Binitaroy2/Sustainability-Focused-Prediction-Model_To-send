# src/streamlit_app.py
import os
import subprocess
import importlib
import sys

# Import models module
from api.main import scaler, rf_model, cnn, rnn  # api/main.py (subfolder)

# Streamlit app
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Project root for paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "updated_energy_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # Reload the module to pick up new models after retrain
    importlib.reload(sys.modules['api.main'])
    from api.main import scaler, rf_model, cnn, rnn
    return scaler, rf_model, cnn, rnn

def main():
    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability-Focused Energy Predictor")
    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            # Run train.py via subprocess to avoid import execution
            train_path = os.path.join(PROJECT_ROOT, 'src', 'train.py')
            src_dir = os.path.join(PROJECT_ROOT, 'src')
            result = subprocess.run(['python', train_path], capture_output=True, text=True, cwd=src_dir)
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
    # Predictions
    scaler_, rf_, cnn_, rnn_ = load_models()
    X = df.drop(columns=["Target"])
    y = df["Target"]
    Xs = scaler_.transform(X)
    preds = rf_.predict(Xs)
    st.markdown("### RF: Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y, preds, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (RF)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()