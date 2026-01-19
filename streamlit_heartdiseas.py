import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# Load pre-trained model and scaler
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('heart_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for accuracy display
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)

# Streamlit UI
st.title("🩺 Heart Disease Prediction App")

st.write("Enter patient details below:")

# Input fields for patient data
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0, 1])  # 0 = female, 1 = male
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

# Prediction button
if st.button("Predict"):
    # Prepare input
    patient_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
    patient_data = scaler.transform(patient_data)

    prediction = model.predict(patient_data)

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")

# Show model accuracy
st.write("Model Accuracy on Test Data:", accuracy_score(y_test, model.predict(X_test_scaled)))
