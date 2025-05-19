### 🔍 Bank Consumer Churn Prediction

This project predicts bank customer churn using three classification models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

It uses MLflow to:
- Track experiments
- Log models and metrics
- Save preprocessing pipelines
- Store evaluation artifacts like confusion matrices

### 📦 Requirements

Install the required Python libraries using:

pip install -r requirements.txt

📁 Dataset

The dataset used is Churn_Modelling.csv, which should be placed inside a folder named dataset:

project_root/
│
├── dataset/
│   └── Churn_Modelling.csv
└── src/
    └── train.py

### 🚀 Running the Project

Make sure the MLflow tracking server is running locally:

mlflow ui

Then, execute the training script from the src directory:

python src/train.py

This will:
- Preprocess and rebalance the dataset
- Train and evaluate all three models
- Log experiments, metrics, and artifacts to MLflow

### 📊 Viewing Results

Once the script finishes:
1. Open your browser and go to: http://127.0.0.1:5000
2. Use the MLflow UI to:
   - Compare model performance
   - View logged confusion matrices
   - Check parameters and metrics
   - Download trained models and preprocessing artifacts
