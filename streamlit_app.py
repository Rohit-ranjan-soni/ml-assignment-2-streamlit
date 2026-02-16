
import streamlit as st
import pandas as pd
import joblib, json
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="ML Assignment 2 - Classification Models", layout="wide")

st.title("ML Assignment 2: Multiple Classification Models (Interactive Demo)")

# Load metrics + artifacts
metrics_df = pd.read_csv("metrics.csv")
with open("artifacts.json", "r") as f:
    artifacts = json.load(f)

model_names = metrics_df["model"].tolist()

st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox("Select a model", model_names)

# Load model
model = joblib.load(f"model/{selected_model}.joblib")

st.subheader("1) Evaluation Metrics (on held-out test split)")
row = metrics_df[metrics_df["model"] == selected_model].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{row['accuracy']:.4f}")
c2.metric("AUC", f"{row['auc']:.4f}")
c3.metric("MCC", f"{row['mcc']:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("Precision", f"{row['precision']:.4f}")
c5.metric("Recall", f"{row['recall']:.4f}")
c6.metric("F1", f"{row['f1']:.4f}")

st.subheader("2) Confusion Matrix / Classification Report (on test split)")
cm = artifacts[selected_model]["confusion_matrix"]
st.write("Confusion Matrix (rows=true, cols=pred):")
st.dataframe(pd.DataFrame(cm, columns=["Pred 0","Pred 1"], index=["True 0","True 1"]))

st.write("Classification Report:")
st.code(artifacts[selected_model]["classification_report_text"])

st.subheader("3) Upload a CSV to Predict")
st.write("Upload a CSV containing ONLY feature columns (same as training).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    # Predict
    preds = model.predict(df)
    st.write("Predictions (0/1):")
    st.dataframe(pd.DataFrame({"prediction": preds}))

    # Probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[:, 1]
        st.write("Prediction Probability (class=1):")
        st.dataframe(pd.DataFrame({"prob_class_1": proba}))
