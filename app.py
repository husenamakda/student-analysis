
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("Student Performance Predictor")

try:
    # Load dataset
    df = pd.read_csv("StudentsPerformance.csv")

    # Inputs
    gender = st.selectbox("Gender", df["gender"].unique())
    lunch = st.selectbox("Lunch", df["lunch"].unique())
    prep = st.selectbox("Test Preparation", df["test preparation course"].unique())

    input_data = pd.DataFrame({
        "gender": [gender],
        "lunch": [lunch],
        "test preparation course": [prep]
    })

    # Prepare data
    X = df.drop("math score", axis=1)
    y = df["math score"]

    X = pd.get_dummies(X)
    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Model
    model = RandomForestRegressor()
    model.fit(X, y)

    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Math Score: {prediction[0]:.2f}")

except Exception as e:
    st.error(f"Error: {e}")