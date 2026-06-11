import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="XGBoost M&V", layout="wide")
st.title("🏠 XGBoost M&V Dashboard")

@st.cache_resource
def load():
    model = joblib.load('models/model.pkl')
    with open('models/features.txt', 'r') as f:
        features = [l.strip() for l in f.readlines()]
    return model, features

model, FEATURES = load()
st.success("✅ XGBoost ready!")

# Sidebar
st.sidebar.header("Parameters")
temp = st.sidebar.slider("Temperature (°C)", 22, 35, 28)
hum = st.sidebar.slider("Humidity (%)", 60, 95, 80)
hour = st.sidebar.slider("Hour", 0, 23, 14)
dow = st.sidebar.selectbox("Day", list(range(7)), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.sidebar.selectbox("Month", list(range(1,13)))
area = st.sidebar.number_input("Floor Area (m²)", 60, 300, 120)
occ = st.sidebar.number_input("Occupants", 1, 8, 4)
retro = st.sidebar.selectbox("Retrofit", [0,1], format_func=lambda x: "Yes" if x else "No")

# Feature engineering
hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)
month_sin = np.sin(2*np.pi*month/12)
month_cos = np.cos(2*np.pi*month/12)
is_weekend = 1 if dow >= 5 else 0
temp_humid = temp * hum / 100
occ_area = occ / area

X_input = pd.DataFrame([[
    temp, hum, hour, dow, month, area, occ, retro,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_humid, occ_area
]], columns=FEATURES)

if st.button("Predict"):
    pred = model.predict(X_input)[0] / 10
    st.metric("Energy", f"{pred:.2f} kWh")
    
    if retro == 1:
        X_base = X_input.copy()
        X_base['retrofit'] = 0
        base = model.predict(X_base)[0] / 10
        save = base - pred
        st.success(f"Savings: {save:.2f} kWh ({save/base*100:.1f}%)")
        
        fig, ax = plt.subplots()
        ax.bar(['Baseline', 'Retrofitted'], [base, pred], color=['red','green'])
        ax.set_ylabel('kWh')
        st.pyplot(fig)
