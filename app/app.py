import streamlit as st
import pandas as pd
import joblib
import os


# Load trained model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgboost.pkl")

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Employee Retention Prediction",
    layout="centered"
)

st.title("Employee Retention Prediction App")
st.write("Predict whether an employee is likely to look for a job change.")


# User Input Section


# System / Identifier Fields (Hidden from user)

enrollee_id = 1
city = 1

# Demographic Information

st.markdown("### Demographic Information")

city_development_index = st.slider(
    "City Development Index",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)


# Education & Experience

st.markdown("### Education & Experience")

relevent_experience = st.selectbox(
    "Relevant Experience",
    ["Has relevent experience", "No relevent experience"]
)

enrolled_university = st.selectbox(
    "Enrolled University",
    ["no_enrollment", "Full time course", "Part time course"]
)

education_level = st.selectbox(
    "Education Level",
    ["Primary School", "High School", "Graduate", "Masters", "Phd"]
)

major_discipline = st.selectbox(
    "Major Discipline",
    ["STEM", "Arts", "Business Degree", "Humanities", "Other"]
)

experience = st.selectbox(
    "Total Experience (Years)",
    ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10+", "20+"]
)

# Employment Details

st.markdown("### Employment Details")

company_size = st.selectbox(
    "Company Size",
    ["<10", "10-49", "50-99", "100-500", "500-999",
     "1000-4999", "5000-9999", "10000+"]
)

company_type = st.selectbox(
    "Company Type",
    ["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Other"]
)

last_new_job = st.selectbox(
    "Years Since Last Job Change",
    ["never", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
)

training_hours = st.number_input(
    "Training Hours",
    min_value=0,
    max_value=500,
    value=50
)

# Prediction

if st.button("Predict Job Change"):

    input_dict = {
        "enrollee_id": enrollee_id,
        "city": city,
        "city_development_index": city_development_index,
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "last_new_job": last_new_job,
        "training_hours": training_hours
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Employee is LIKELY to change job (Probability: {probability:.2f})")
    else:
        st.success(f"Employee is UNLIKELY to change job (Probability: {probability:.2f})")
