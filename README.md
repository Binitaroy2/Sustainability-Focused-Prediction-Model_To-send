# 🌱 Sustainability-Focused Energy Consumption Prediction Project

A modern MLOps project to predict energy consumption based on sustainability-related features, leveraging machine learning, deep learning, and an interactive web UI.  
Built for transparency, reproducibility, and easy deployment!

---

## 📂 Repository Structure

```
Sustainability-Focused Energy Consumption Prediction Project/
│
├── data/
│   └── updated_energy_dataset.csv
│
├── models/
│   ├── best_rf_model.pkl         # Trained Random Forest model
│   ├── cnn_model.keras           # Trained CNN (Keras format)
│   ├── rnn_model.keras           # Trained RNN (Keras format)
│   └── scaler.pkl                # Scaler for numeric feature normalization
│
├── src/
│   ├── train.py                  # Training script for all models and MLflow tracking
│   └── api/
│       ├── main.py               # FastAPI app for serving predictions + UI
│       └── ui/
│           ├── index.html        # Web UI (user interface)
│           ├── styles.css        # UI styling
│           └── script.js         # UI JavaScript logic
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- **Languages**: Python 🐍, HTML, CSS, JavaScript
- **Libraries & Frameworks**:
  - **Data Processing**: pandas, numpy
  - **Visualization**: matplotlib, seaborn, plotly
  - **Machine Learning**: scikit-learn
  - **Deep Learning**: TensorFlow, Keras
  - **Model Deployment**: FastAPI, Uvicorn
  - **MLOps & Experiment Tracking**: MLflow

---

## 🚀 How to Run

### **1. Clone the Repository**
```sh
git clone https://github.com/binita-roy/Sustainability-Focused-Energy-Consumption-Prediction-Project.git
cd Sustainability-Focused-Energy-Consumption-Prediction-Project
```

### **2. Install Requirements**
```sh
pip install -r requirements.txt
```

### **3. Train Models and Prepare Artifacts**
```sh
cd src
python train.py
```
This will:
- Train Random Forest, CNN, and RNN models
- Log experiments with MLflow
- Save model files and scaler in `/models/`

### **4. Run the Web Application (API + UI)**
```sh
cd api
uvicorn main:app --reload
```
Open your browser at:  
[http://127.0.0.1:8000/ui/index.html](http://127.0.0.1:8000/ui/index.html)

---

## 🌐 Application Overview

- **Web UI**: User-friendly form for entering energy project details, with dropdowns for categorical features and selection of the prediction model (Random Forest, CNN, RNN).
- **API**: FastAPI backend for real-time predictions, serving both the UI and the ML/DL models.
- **Feature Selection**: Only the most relevant predictors are used for improved accuracy and interpretability.

---

## 📊 Features Used for Prediction

- Energy_Production_MWh
- Type_of_Renewable_Energy (Solar, Wind, Hydroelectric, etc.)
- Installed_Capacity_MW
- Energy_Storage_Capacity_MWh
- Storage_Efficiency_Percentage
- Grid_Integration_Level

---

## 🔬 MLflow Experiment Tracking

All model training runs and metrics are logged using [MLflow](https://mlflow.org/).

To view the MLflow UI locally:
```sh
mlflow ui
```
Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🚢 Deployment

You can deploy this application on **Azure App Service** or any cloud/server that supports Python 3.8+, FastAPI, and Uvicorn.

**Quick Azure steps:**
1. Create an Azure App Service (Linux, Python 3.8+)
2. Deploy code via GitHub Actions or Zip Deploy
3. Set the startup command to:
   ```
   gunicorn -w 1 -k uvicorn.workers.UvicornWorker src.api.main:app
   ```
4. Visit your Azure Web App URL!

---

## 🤝 Contributions

Open to feedback, issues, and contributions!  
Fork, star, or submit a pull request.

---

## 📄 License

[MIT License](LICENSE)  
Feel free to use, adapt, and share.

---

**Built with ❤️ for sustainable innovation and reproducible MLOps.**
