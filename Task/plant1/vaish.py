import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# ============================
# Load models
# ============================
scaler_X = joblib.load(r"C:/Users/User/Desktop/solar_code/Task/plant1/mod/scaler_X.pkl")
cnn_model = load_model(r"C:/Users/User/Desktop/solar_code/Task/plant1/mod/cnn_model.h5", compile=False)
cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

xgb_model = XGBRegressor()
xgb_model.load_model(r"C:/Users/User/Desktop/solar_code/Task/plant1/mod/xgb_model.json")

# ============================
# Preprocessing
# ============================
def preprocess_input(df_input, sequence_length=6):
    df_input['DATE_TIME'] = pd.to_datetime(df_input['DATE_TIME'])
    df_input['hour'] = df_input['DATE_TIME'].dt.hour
    df_input['minute'] = df_input['DATE_TIME'].dt.minute
    df_input['time_fraction'] = df_input['hour'] + df_input['minute'] / 60.0
    df_input['sin_time'] = np.sin(2 * np.pi * df_input['time_fraction'] / 24)
    df_input['cos_time'] = np.cos(2 * np.pi * df_input['time_fraction'] / 24)
    df_input = df_input.drop(columns=['hour','minute','time_fraction','DATE_TIME'])

    X_seq = [df_input.iloc[:sequence_length].values]
    X_seq = np.array(X_seq)
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_scaled = scaler_X.transform(X_flat).reshape(X_seq.shape)
    return X_scaled

# ============================
# Prediction
# ============================
def predict_ac_power(new_data):
    X_scaled = preprocess_input(new_data)
    cnn_features = cnn_feature_extractor.predict(X_scaled)
    y_pred = xgb_model.predict(cnn_features)
    return float(y_pred[-1])

# ============================
# Streamlit Layout
# ============================
st.set_page_config(page_title="Solar Power Prediction", page_icon="⚡", layout="wide")

# Layout: left sidebar + main area
with st.sidebar:
    st.markdown(
        """
        <h1 style='font-size:26px; color:#ffdd00;'>⚡ Solar Power Forecasting</h1>
        <p style='font-size:16px; color:#cccccc;'>Select the date & time to start predictions.</p>
        """,
        unsafe_allow_html=True
    )

    base_date = st.date_input("📅 **Select Date**", datetime(2020, 5, 15))
    base_time = st.time_input("⏰ **Start Time**", datetime(2020, 5, 15, 6, 0).time())


# Main content
st.markdown("<h2 style='text-align: center; color:#00f5d4;'>🌞 Provide Environmental Inputs</h2>", unsafe_allow_html=True)

# Generate 6 timestamps
base_datetime = datetime.combine(base_date, base_time)
timestamps = [base_datetime + timedelta(minutes=15*i) for i in range(6)]

rows = []
for i, ts in enumerate(timestamps):
    col1, col2, col3 = st.columns(3)
    with col1:
        amb = st.number_input(f"Ambient Temp {i+1} (°C)", value=25.0, step=0.1, key=f"amb_{i}")
    with col2:
        mod = st.number_input(f"Module Temp {i+1} (°C)", value=30.0, step=0.1, key=f"mod_{i}")
    with col3:
        irr = st.number_input(f"Irradiation {i+1}", value=0.2, step=0.01, key=f"irr_{i}")
    rows.append([ts, amb, mod, irr])

# Predict button
if st.button("🔮 Predict AC Power"):
    sample_data = pd.DataFrame(rows, columns=["DATE_TIME","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"])
    try:
        prediction = predict_ac_power(sample_data)
        next_time = timestamps[-1] + timedelta(minutes=15)

        st.markdown(
            f"""
            <div style="padding:20px; background-color:#1e1e2f; border-radius:10px; text-align:center;">
                <h3 style="color:#ffdd00;">⚡ Prediction Result</h3>
                <p style="color:white; font-size:18px;">
                    Next Interval: <b>{next_time.strftime('%Y-%m-%d %H:%M')}</b><br>
                    Predicted AC Power: <b style="color:#00f5d4;">{prediction:.2f} kW</b>
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error: {e}")
