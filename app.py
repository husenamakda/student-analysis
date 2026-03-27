import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("Student Performance Predictor")

gender = st.selectbox("Gender", ["male", "female"])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
prep = st.selectbox("Test Preparation", ["none", "completed"])

input_data = pd.DataFrame({
    "gender": [gender],
    "lunch": [lunch],
    "test preparation course": [prep]
})

df = pd.read_csv("StudentsPerformance.csv")

X = df.drop("math score", axis=1)
y = df["math score"]

X = pd.get_dummies(X)
input_data = pd.get_dummies(input_data)

input_data = input_data.reindex(columns=X.columns, fill_value=0)

model = RandomForestRegressor()
model.fit(X, y)

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Math Score: {prediction[0]:.2f}")