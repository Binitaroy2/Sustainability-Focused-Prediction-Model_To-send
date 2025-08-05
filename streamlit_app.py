# src/streamlit_app.py
import os
import sys
# No need for path manipulations or chdir

# Import your modules with the 'src.' prefix
from src.train import train_and_log  # src/train.py
from src.api.main import scaler, rf_model, cnn, rnn  # src/api/main.py

# Now the Streamlit app
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Assuming CWD is project root; adjust if needed
PROJECT_ROOT = os.getcwd()
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