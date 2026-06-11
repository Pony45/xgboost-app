import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# Try to import joblib, if fail, use pickle as fallback
try:
    import joblib
except ImportError:
    import pickle as joblib

st.set_page_config(page_title="XGBoost M&V Dashboard", layout="wide")
st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Predict energy savings using **XGBoost** Regressor*")

# ==========================================
# LOAD XGBOOST MODEL
# ==========================================
@st.cache_resource
def load_model():
    possible_paths = [
        'models/model.pkl',
        'models/xgboost_model.pkl',
        'model.pkl',
        'xgboost_model.pkl'
    ]
    
    model = None
    model_path_used = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                model_path_used = path
                break
            except:
                continue
    
    if model is None:
        st.error("❌ XGBoost model not found!")
        return None, None
    
    # Load features
    features_paths = [
        'models/features.txt',
        'models/xgboost_features.txt',
        'features.txt',
        'xgboost_features.txt'
    ]
    
    features = None
    for path in features_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            break
    
    if features is None:
        features = [
            'temperature', 'humidity', 'hour', 'dayofweek', 'month',
            'floor_area', 'occupants', 'retrofit',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'is_weekend', 'temp_humidity', 'occ_per_area'
        ]
    
    st.success(f"✅ XGBoost model loaded!")
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

# ==========================================
# LOAD METRICS
# ==========================================
@st.cache_resource
def load_metrics():
    metrics_paths = [
        'models/metrics.json',
        'models/xgboost_metrics.json',
        'metrics.json',
        'xgboost_metrics.json'
    ]
    
    for path in metrics_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                continue
    return None

metrics = load_metrics()

# ==========================================
# SCALING & CONVERSION
# ==========================================
SCALING_FACTOR = 25

def scale_prediction(prediction):
    return prediction / SCALING_FACTOR

def convert_energy_unit(prediction_kwh, target_unit):
    if target_unit == "Per Hour (kWh)":
        return prediction_kwh, "kWh"
    elif target_unit == "Per Day (kWh)":
        return prediction_kwh * 24, "kWh/day"
    elif target_unit == "Per Month (kWh)":
        return prediction_kwh * 24 * 30, "kWh/month"
    elif target_unit == "Per Year (kWh)":
        return prediction_kwh * 24 * 365, "kWh/year"

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📋 Building Parameters")

# Model performance
st.sidebar.markdown("---")
st.sidebar.subheader("📊 XGBoost Performance")

if metrics:
    with st.sidebar.expander("Performance Metrics", expanded=True):
        r2 = metrics.get('r2', metrics.get('r2_score', 0))
        mae = metrics.get('mae', 0)
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.2f} kWh")
        st.progress(min(r2, 1.0), text=f"Accuracy: {r2*100:.1f}%")
else:
    with st.sidebar.expander("Performance Metrics", expanded=True):
        st.info("Metrics file not found - model still works!")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Display Settings")

unit_option = st.sidebar.selectbox(
    "Energy Unit Display",
    ["Per Hour (kWh)", "Per Day (kWh)", "Per Month (kWh)", "Per Year (kWh)"]
)

st.sidebar.markdown("---")

# Input parameters (Malaysia range)
temp = st.sidebar.slider("🌡️ Temperature (°C)", 18, 40, 28)
humidity = st.sidebar.slider("💧 Humidity (%)", 60, 95, 80)
hour = st.sidebar.slider("⏰ Hour of Day", 0, 23, 14)
dayofweek = st.sidebar.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.sidebar.selectbox("📆 Month", list(range(1,13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", 60, 300, 120)
occupants = st.sidebar.number_input("👥 Occupants", 1, 8, 4)
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0,1], format_func=lambda x: "✅ Yes (Retrofitted)" if x else "❌ No (Baseline)")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if dayofweek >= 5 else 0
temp_humidity = temp * humidity / 100
occ_per_area = occupants / floor_area

features_df = pd.DataFrame([[
    temp, humidity, hour, dayofweek, month, floor_area, occupants, retrofit,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_humidity, occ_per_area
]], columns=FEATURES)

# ==========================================
# MAIN CONTENT
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Energy Prediction", "📈 Savings Analysis", "🔍 Feature Impact"])

# ===== TAB 1: Energy Prediction =====
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Predict Energy (XGBoost)", type="primary", use_container_width=True):
            raw_pred = model.predict(features_df)[0]
            pred = scale_prediction(raw_pred)
            converted, unit = convert_energy_unit(pred, unit_option)
            
            st.subheader("📊 Prediction Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("⚡ Predicted Energy", f"{converted:.2f} {unit}")
            
            if retrofit == 1:
                base_df = features_df.copy()
                base_df['retrofit'] = 0
                raw_base = model.predict(base_df)[0]
                base_pred = scale_prediction(raw_base)
                
                savings = base_pred - pred
                savings_pct = (savings / base_pred) * 100
                savings_conv, _ = convert_energy_unit(savings, unit_option)
                
                m2.metric("💰 Savings", f"{savings_conv:.2f} {unit}", delta=f"{savings_pct:.1f}%")
                m3.metric("🏆 Reduction", f"{savings_pct:.1f}%", delta="Good!")
                
                st.success(f"💡 XGBoost: Retrofit saves {savings_conv:.2f} {unit} ({savings_pct:.1f}%)")
                
                # Monthly bill savings
                tariff = 0.52
                monthly_savings = savings * 24 * 30
                monthly_rm = monthly_savings * tariff
                st.info(f"💰 **Estimated monthly bill savings:** RM {monthly_rm:.2f}/month")
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(8,5))
                bars = ax.bar(['Baseline\n(No Retrofit)', 'Retrofitted'], 
                             [base_pred, pred], color=['#e74c3c', '#2ecc71'], edgecolor='black')
                for bar, val in zip(bars, [base_pred, pred]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{val:.2f} kWh', ha='center', fontweight='bold')
                ax.set_ylabel('Energy Consumption (kWh)')
                ax.set_title('XGBoost: Retrofit Impact', fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                
                # Gauge chart
                fig2, ax2 = plt.subplots(figsize=(8,2.5))
                color = '#2ecc71' if savings_pct > 20 else '#f39c12' if savings_pct > 10 else '#e74c3c'
                label = 'High' if savings_pct > 20 else 'Medium' if savings_pct > 10 else 'Low'
                ax2.barh([0], [min(savings_pct,100)], color=color, height=0.4, edgecolor='black')
                ax2.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
                ax2.set_xlim(0,100)
                ax2.set_yticks([])
                ax2.set_xlabel('Savings (%)')
                ax2.set_title(f'Efficiency: {label} ({savings_pct:.1f}% savings)', fontweight='bold')
                ax2.text(savings_pct + 2, 0, f'{savings_pct:.1f}%', va='center', fontweight='bold')
                st.pyplot(fig2)
                
            else:
                retro_df = features_df.copy()
                retro_df['retrofit'] = 1
                raw_retro = model.predict(retro_df)[0]
                retro_pred = scale_prediction(raw_retro)
                potential = pred - retro_pred
                potential_pct = (potential / pred) * 100
                potential_conv, _ = convert_energy_unit(potential, unit_option)
                
                m2.metric("💰 Potential Savings", f"{potential_conv:.2f} {unit}", delta=f"{potential_pct:.1f}%")
                m3.metric("🏆 Would Save", f"{potential_pct:.1f}%", delta="If retrofitted")
                st.info(f"💡 XGBoost: If retrofitted, save ~{potential_conv:.2f} {unit} ({potential_pct:.1f}%)")
                
                fig_simple, ax_simple = plt.subplots(figsize=(8,5))
                ax_simple.bar(['Current\n(No Retrofit)', 'If Retrofitted'], 
                             [pred, retro_pred], color=['#e74c3c', '#2ecc71'], edgecolor='black')
                for i, v in enumerate([pred, retro_pred]):
                    ax_simple.text(i, v + 0.05, f'{v:.2f} kWh', ha='center', fontweight='bold')
                ax_simple.set_ylabel('Energy (kWh)')
                ax_simple.set_title('XGBoost: Potential Retrofit Impact', fontweight='bold')
                ax_simple.grid(axis='y', alpha=0.3)
                st.pyplot(fig_simple)
    
    with col2:
        st.info("""
        **📖 About XGBoost M&V System**
        
        🎯 **Model:** XGBoost Regressor
        
        📊 **Features (15):**
        - Weather (temp, humidity)
        - Time (hour, day, month)
        - Building (area, occupants)
        - Retrofit status
        
        🏠 **Malaysia Context:**
        - Scaled for residential homes
        - TNB tariff: ~RM0.52/kWh
        """)

# ===== TAB 2: Savings Analysis =====
with tab2:
    st.subheader("💰 Retrofit Savings Analysis")
    
    # Calculate for current parameters
    base_df = features_df.copy()
    base_df['retrofit'] = 0
    retro_df = features_df.copy()
    retro_df['retrofit'] = 1
    
    raw_base = model.predict(base_df)[0]
    raw_retro = model.predict(retro_df)[0]
    
    base_pred = scale_prediction(raw_base)
    retro_pred = scale_prediction(raw_retro)
    
    savings = base_pred - retro_pred
    savings_pct = (savings / base_pred) * 100
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🏚️ Baseline (No Retrofit)", f"{base_pred:.2f} kWh")
    col_b.metric("🏠 Retrofitted", f"{retro_pred:.2f} kWh", delta=f"-{savings:.2f} kWh")
    col_c.metric("💵 Energy Savings", f"{savings:.2f} kWh", delta=f"{savings_pct:.1f}%")
    
    # Financial impact
    tariff = st.number_input("Electricity Price (RM/kWh)", min_value=0.10, max_value=1.50, value=0.52, step=0.01)
    hourly_savings_rm = savings * tariff
    daily_savings_rm = hourly_savings_rm * 24
    monthly_savings_rm = daily_savings_rm * 30
    yearly_savings_rm = monthly_savings_rm * 12
    
    st.info(f"""
    💰 **Financial Impact (at RM {tariff}/kWh):**
    - Daily savings: **RM {daily_savings_rm:.2f}**
    - Monthly savings: **RM {monthly_savings_rm:.2f}**
    - Yearly savings: **RM {yearly_savings_rm:.2f}**
    """)
    
    # Comparison chart
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(['Baseline\n(No Retrofit)', 'Retrofitted'], [base_pred, retro_pred],
                  color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, [base_pred, retro_pred]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f} kWh', ha='center', fontweight='bold')
    ax.annotate(f'💡 Savings: {savings:.2f} kWh\n({savings_pct:.1f}%)',
                xy=(1, retro_pred + savings/2), xytext=(1.4, retro_pred + savings/2 + 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2), fontsize=10, fontweight='bold')
    ax.set_ylabel('Energy Consumption (kWh)')
    ax.set_title('XGBoost: Retrofit Impact Analysis', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

# ===== TAB 3: Feature Impact =====
with tab3:
    st.subheader("🔍 Feature Impact Analysis")
    
    feature_to_vary = st.selectbox(
        "Select feature to analyze:",
        ['temperature', 'humidity', 'hour', 'occupants', 'floor_area']
    )
    
    if feature_to_vary == 'temperature':
        x_range = np.arange(22, 36, 1)
        x_label = "Temperature (°C)"
        fixed_values = {'temperature': 28, 'humidity': 80, 'hour': 14, 'occupants': 4, 'floor_area': 120}
    elif feature_to_vary == 'humidity':
        x_range = np.arange(60, 96, 5)
        x_label = "Humidity (%)"
        fixed_values = {'temperature': 28, 'humidity': 80, 'hour': 14, 'occupants': 4, 'floor_area': 120}
    elif feature_to_vary == 'hour':
        x_range = np.arange(0, 24, 1)
        x_label = "Hour of Day"
        fixed_values = {'temperature': 28, 'humidity': 80, 'hour': 14, 'occupants': 4, 'floor_area': 120}
    elif feature_to_vary == 'occupants':
        x_range = np.arange(1, 9, 1)
        x_label = "Number of Occupants"
        fixed_values = {'temperature': 28, 'humidity': 80, 'hour': 14, 'occupants': 4, 'floor_area': 120}
    else:
        x_range = np.arange(60, 301, 20)
        x_label = "Floor Area (m²)"
        fixed_values = {'temperature': 28, 'humidity': 80, 'hour': 14, 'occupants': 4, 'floor_area': 120}
    
    predictions_baseline = []
    predictions_retrofit = []
    
    for x_val in x_range:
        fixed_values[feature_to_vary] = x_val
        
        # Feature engineering
        h_sin = np.sin(2 * np.pi * fixed_values['hour'] / 24)
        h_cos = np.cos(2 * np.pi * fixed_values['hour'] / 24)
        m_sin = np.sin(2 * np.pi * 6 / 12)
        m_cos = np.cos(2 * np.pi * 6 / 12)
        temp_humid = fixed_values['temperature'] * fixed_values['humidity'] / 100
        occ_area = fixed_values['occupants'] / fixed_values['floor_area']
        
        # Baseline
        feat_baseline = [[
            fixed_values['temperature'], fixed_values['humidity'], fixed_values['hour'], 0, 6,
            fixed_values['floor_area'], fixed_values['occupants'], 0,
            h_sin, h_cos, m_sin, m_cos, 0, temp_humid, occ_area
        ]]
        pred_baseline = model.predict(pd.DataFrame(feat_baseline, columns=FEATURES))[0] / SCALING_FACTOR
        predictions_baseline.append(pred_baseline)
        
        # Retrofit
        feat_retrofit = [[
            fixed_values['temperature'], fixed_values['humidity'], fixed_values['hour'], 0, 6,
            fixed_values['floor_area'], fixed_values['occupants'], 1,
            h_sin, h_cos, m_sin, m_cos, 0, temp_humid, occ_area
        ]]
        pred_retrofit = model.predict(pd.DataFrame(feat_retrofit, columns=FEATURES))[0] / SCALING_FACTOR
        predictions_retrofit.append(pred_retrofit)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range, predictions_baseline, 'o-', label='Baseline (No Retrofit)', color='#e74c3c', linewidth=2)
    ax.plot(x_range, predictions_retrofit, 's-', label='Retrofitted', color='#2ecc71', linewidth=2)
    ax.fill_between(x_range, predictions_baseline, predictions_retrofit, alpha=0.3, color='blue', label='Potential Savings')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_title(f'Impact of {feature_to_vary.replace("_", " ").title()} on Energy Consumption', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("""
    ### 📝 Interpretation:
    - **Red line**: Energy if NOT retrofitted
    - **Green line**: Energy AFTER retrofit
    - **Blue area**: Energy savings from retrofit
    - **Bigger gap** = Retrofit more effective
    """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 XGBoost M&V System | Thesis Project | Scaled for Malaysian Residential Buildings</p>
    <p>📌 TNB Tariff: ~RM0.52/kWh | Typical home: 300-600 kWh/month</p>
</div>
""", unsafe_allow_html=True)
