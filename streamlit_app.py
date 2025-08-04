# streamlit.py
import os
import sys

# 1) Ensure src/ is on the path so we can import train & api
BASE = os.path.dirname(__file__)
SRC  = os.path.join(BASE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 2) Import your existing modules
from train import train_and_log       # src/train.py
from api.main import scaler, rf_model, cnn, rnn  # src/api/main.py loads these

# 3) Data path (adjust if yours differs)
DATA_PATH = os.path.join(BASE, "data", "updated_energy_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # scaler, rf_model, cnn, rnn were already loaded at import-time in main.py
    return scaler, rf_model, cnn, rnn

def main():
    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability-Focused Energy Predictor")

    # ── Sidebar controls ──
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Retraining via train.py…"):
            train_and_log()
        st.success("✅ Re-training done. Click ‘Reload Models’ below.")

    if st.sidebar.button("Reload Models"):
        load_models.clear()  # drop cache so we re-import artifacts
        st.success("🔄 Models reloaded.")

    # ── Main panel ──
    df = load_data()
    st.markdown("### Raw data preview")
    st.dataframe(df.head())

    scaler_, rf_, cnn_, rnn_ = load_models()

    # Example: RF Actual vs Predicted
    X = df.drop(columns=["Target"])
    y = df["Target"]
    Xs = scaler_.transform(X)
    preds = rf_.predict(Xs)

    fig, ax = plt.subplots()
    ax.scatter(y, preds, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (RF)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
