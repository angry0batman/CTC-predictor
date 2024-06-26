import streamlit as st
import pandas as pd
import joblib

# Load the model and column names
model = joblib.load('ctc_model.pkl')
column_names = joblib.load('column_names.pkl')

# Function to predict CTC
def predict_ctc(experience, role, location, tier_of_college, soft_skills):
    data = {
        'Experience': [experience],
        'Role': [role],
        'Location': [location],
        'Tier of College': [tier_of_college],
        'Soft skills': [soft_skills]
    }
    df = pd.get_dummies(pd.DataFrame(data))
    
    # Ensure the same columns as training data
    for col in column_names:
        if col not in df.columns:
            df[col] = 0
    df = df[column_names]
    
    return model.predict(df)[0]

# Streamlit app
st.title("CTC Prediction")

experience = st.selectbox("Experience", [1, 2])
role = st.selectbox("Role", ["SDE", "DA", "DS", "Manager"])
location = st.selectbox("Location", ["Remote", "Onsite"])
tier_of_college = st.selectbox("Tier of College", [1, 2])
soft_skills = st.slider("Soft skills", 60, 90, 70)

if st.button("Predict CTC"):
    ctc = predict_ctc(experience, role, location, tier_of_college, soft_skills)
    st.write(f"Predicted CTC: {ctc}")

st.markdown("""
<style>
    .main {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

if st.button("Return to Main Page"):
    st.write("Redirecting to main page...")
