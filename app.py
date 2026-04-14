import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Ride Demand Predictor",
    page_icon="🚖",
    layout="centered"
)

# ------------------------------
# LOAD MODEL (CACHE)
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ------------------------------
# FEATURE ENGINEERING FUNCTION
# ------------------------------
def create_features(dt):
    return pd.DataFrame([{
        "hour_of_day": dt.hour,
        "day_of_week": dt.dayofweek,
        "day": dt.day,
        "month": dt.month,
        "is_weekend": 1 if dt.dayofweek in [5, 6] else 0
    }])

# ------------------------------
# UI HEADER
# ------------------------------
st.title("🚖 Ride Demand Prediction")
st.markdown(
    "Predict the number of rides based on **date and time** using a trained ML model."
)

st.divider()

# ------------------------------
# INPUT SECTION
# ------------------------------
st.subheader("📅 Select Date & Time")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date")

with col2:
    time = st.time_input("Time")

# ------------------------------
# PREDICTION BUTTON
# ------------------------------
if st.button("🔮 Predict Ride Demand", use_container_width=True):

    try:
        # Combine date & time
        dt = pd.to_datetime(f"{date} {time}")

        # Create features
        X = create_features(dt)

        # Predict
        prediction = model.predict(X)[0]

        # Display result
        st.success(f"🚖 Predicted Ride Demand: {round(prediction, 2)} rides")

        # Extra insights
        st.markdown("### 📊 Input Breakdown")
        st.write({
            "Hour": dt.hour,
            "Day of Week": dt.dayofweek,
            "Month": dt.month,
            "Weekend": "Yes" if dt.dayofweek in [5, 6] else "No"
        })

    except Exception as e:
        st.error(f"Error: {e}")

# ------------------------------
# SIDEBAR INFO
# ------------------------------
st.sidebar.title("ℹ️ About")

st.sidebar.markdown("""
**Ride Demand Prediction App**

- Model: Random Forest  
- Features:
  - Hour of Day  
  - Day of Week  
  - Day  
  - Month  
  - Weekend Flag  

---

Built as an end-to-end ML project:
- Data Processing  
- Feature Engineering  
- Model Training  
- Deployment with Streamlit  
""")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built by Aditya 🚀 | Data Science Project")
