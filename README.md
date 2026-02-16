
# ML Assignment 2 — Multiple Classification Models + Streamlit Deployment

## 1. Problem Statement
The objective of this assignment is to:
- Implement multiple classification models
- Evaluate them using standard metrics
- Build an interactive Streamlit web application
- Deploy the application on Streamlit Community Cloud

This demonstrates a complete real-world ML workflow:
Data → Modeling → Evaluation → UI → Deployment.

---

## 2. Dataset Description

Dataset Used: **Breast Cancer Wisconsin (Diagnostic)**  
(Source: scikit-learn built-in dataset)

- Total Samples: 569
- Total Features: 30 numeric features
- Target Type: Binary Classification (0 = Malignant, 1 = Benign)

This satisfies assignment constraints:
- More than 500 rows
- More than 12 features

---

## 3. Models Implemented

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest  
6. XGBoost  

Tree-based and linear models were both evaluated to compare performance across different learning paradigms.

---

## 4. Evaluation Metrics Used

Each model was evaluated using:

- Accuracy  
- AUC (ROC Area Under Curve)  
- Precision  
- Recall  
- F1-Score  
- MCC (Matthews Correlation Coefficient)

These metrics provide a comprehensive evaluation beyond just accuracy.

---

## 5. Model Comparison

All model metrics are stored in `metrics.csv` and displayed interactively in the Streamlit app.

The best-performing models achieved:

- Accuracy ≈ 97%
- AUC ≈ 0.98+
- High F1 and MCC scores

---

## 6. Streamlit Application Features

The deployed Streamlit application includes:

- Model selection dropdown
- Real-time metric display
- Confusion matrix visualization
- Classification report display
- CSV upload for new predictions
- Probability output (if supported by model)

---

## 7. How to Run Locally

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run streamlit_app.py

---

## 8. Repository Structure

ml-assignment-2-streamlit/
│
├── streamlit_app.py
├── requirements.txt
├── runtime.txt
├── metrics.csv
├── artifacts.json
├── README.md
└── model/
├── LogisticRegression.joblib
├── DecisionTree.joblib
├── KNN.joblib
├── NaiveBayes.joblib
├── RandomForest.joblib
└── XGBoost.joblib


---

## 9. Deployment Links

GitHub Repository:
https://github.com/Rohit-ranjan-soni/ml-assignment-2-streamlit/

Live Streamlit Application:
https://ml-assignment-2-app-upwqdwhtxrnfh8kkb5jza5.streamlit.app/

---

## 10. Observations

- Logistic Regression and KNN benefit significantly from feature scaling.
- Tree-based models capture non-linear patterns effectively.
- Random Forest and XGBoost provide strong performance with robustness.
- MCC is a reliable metric for balanced binary classification.
- Deployment demonstrates end-to-end ML system design beyond just modeling.

---

## 11. Deployment Environment

- Python 3.10
- Streamlit Community Cloud
- scikit-learn, pandas, numpy, xgboost

---
