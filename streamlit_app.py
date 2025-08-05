# src/streamlit_app.py

import os
import sys

# ─── 1) Determine absolute paths ───
THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(THIS_FILE)                     # .../<repo>/src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                       # .../<repo>

# ─── 2) Temporarily switch cwd into src/ ───
os.chdir(SRC_DIR)

# ─── 3) Make sure Python can import from src/ ───
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ─── 4) Import your untouched modules ───
from train import train_and_log             # src/train.py
from api.main import scaler, rf_model, cnn, rnn  # src/api/main.py

# ─── 5) Restore cwd back to project root ───
os.chdir(PROJECT_ROOT)


# ─── 6) Now the Streamlit app ───
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "updated_energy_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # these were loaded when api.main was imported
    return scaler, rf_model, cnn, rnn

def main():
    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability-Focused Energy Predictor")

    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            train_and_log()
        st.success("✅ Model retrained! Click ‘Reload Models’ to pick up changes.")

    if st.sidebar.button("Reload Models"):
        load_models.clear()
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
