import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="CLV Prediction", layout="wide")

st.title("📊 Customer Lifetime Value Prediction App")

# Load dataset
data = pd.read_csv("customer_data.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Features & target
X = data[['Age', 'Annual_Income', 'Spending_Score']]
y = data['CLV']

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Input UI
st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 70)
income = st.number_input("Annual Income", value=30000)
score = st.slider("Spending Score", 1, 100)

if st.button("Predict"):
    result = model.predict([[age, income, score]])
    st.success(f"Predicted CLV: ₹{result[0]:.2f}")

# Graph
st.subheader("📈 Income vs CLV")

fig, ax = plt.subplots()
ax.scatter(data['Annual_Income'], data['CLV'])
ax.set_xlabel("Income")
ax.set_ylabel("CLV")

st.pyplot(fig)