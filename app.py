import streamlit as st
import pickle
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Insurance Cost Prediction App")
st.write("Predict medical insurance charges using a trained ML model.")

# ---------------- Load Model & Scaler ----------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# ---------------- User Inputs ----------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox(
    "Region",
    ["Southwest", "Southeast", "Northwest", "Northeast"]
)

# ---------------- Feature Engineering (MATCH TRAINING) ----------------

# is_female
is_female = 1 if sex == "Female" else 0

# is_smoker
is_smoker = 1 if smoker == "Yes" else 0

# region_southeast (only this region was used in training)
region_southeast = 1 if region == "Southeast" else 0

# bmi_category_Obese (BMI >= 30)
bmi_category_Obese = 1 if bmi >= 30 else 0

# ---------------- Scaling (ONLY numeric features) ----------------
# Scaler was trained on: age, bmi, children
numeric_data = np.array([[age, bmi, children]])

# ---------------- Prediction ----------------
if st.button("Predict Insurance Cost 💰"):
    try:
        numeric_scaled = scaler.transform(numeric_data)

        # FINAL INPUT (EXACT ORDER AS TRAINING)
        final_input = np.array([[
            numeric_scaled[0][0],   # age
            is_female,              # is_female
            numeric_scaled[0][1],   # bmi
            numeric_scaled[0][2],   # children
            is_smoker,              # is_smoker
            region_southeast,       # region_southeast
            bmi_category_Obese      # bmi_category_Obese
        ]])

        prediction = model.predict(final_input)

        st.success(f"💵 Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
