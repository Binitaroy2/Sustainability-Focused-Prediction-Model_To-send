# streamlit_app.py

import os
import sys

# ─── 1) Prep Python path & CWD so that train.py's relative read_csv works ───
BASE = os.path.dirname(__file__)
SRC  = os.path.join(BASE, "src")

# 1a) Make sure we can import modules from src/
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# 1b) Change working dir into src/ while importing train so its
#     pd.read_csv("../data/...") points at your data folder correctly
_orig_cwd = os.getcwd()
os.chdir(SRC)

# ─── 2) Import your existing modules ───
from train import train_and_log            # src/train.py
from api.main import scaler, rf_model, cnn, rnn  # src/api/main.py sets these up

# ─── 3) Restore the original working directory ───
os.chdir(_orig_cwd)


# ─── 4) Now the normal Streamlit app ───
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(BASE, "data", "updated_energy_dataset.csv")

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

    # ─ Sidebar controls ─
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            train_and_log()          # uses src/train.py
        st.success("✅ Retrained! Reload models to pick up changes.")

    if st.sidebar.button("Reload Models"):
        load_models.clear()         # drops cache so new artifacts get picked up
        st.success("🔄 Models reloaded.")

    # ─ Data & Predictions ─
    df = load_data()
    st.markdown("### Raw data preview")
    st.dataframe(df.head())

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
