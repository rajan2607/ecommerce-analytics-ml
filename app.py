import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction Dashboard")

# Load dataset
df = pd.read_csv("data/processed/rfm_with_churn_predictions.csv")

st.subheader("Dataset Overview")
st.dataframe(df.head())

# Metrics
st.subheader("Key Metrics")

avg_churn = df["Churn_Probability"].mean()
total_customers = df.shape[0]
high_risk = df[df["Churn_Probability"] > 0.7].shape[0]

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Average Churn Risk", round(avg_churn, 2))
col3.metric("High Risk Customers", high_risk)

# Distribution
st.subheader("Churn Probability Distribution")
st.bar_chart(df["Churn_Probability"])

# Top customers
st.subheader("Top 20 High Risk Customers")

top_customers = df.sort_values(
    "Churn_Probability",
    ascending=False
).head(20)

st.dataframe(top_customers)

# ML prediction
st.subheader("Predict Customer Churn")

recency = st.number_input("Recency")
frequency = st.number_input("Frequency")
monetary = st.number_input("Monetary")

model = joblib.load("model/churn_model.pkl")

if st.button("Predict"):
    prediction = model.predict([[recency, frequency, monetary]])
    probability = model.predict_proba([[recency, frequency, monetary]])[0][1]

    st.write("Prediction:", prediction[0])
    st.write("Churn Probability:", round(probability, 2))