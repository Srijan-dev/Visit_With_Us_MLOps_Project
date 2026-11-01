import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="Wellness Tourism Purchase Predictor", page_icon="🌿", layout="centered")

MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is not available yet. Run the GitHub Actions pipeline "
            "and ensure best_model.pkl is committed to this folder."
        )
    return joblib.load(MODEL_PATH)

st.title("Wellness Tourism Package")
st.subheader("Customer Purchase Prediction")
st.write(
    "Enter the customer's profile details to estimate the likelihood of purchasing "
    "the Wellness Tourism Package."
)

with st.form("customer_details"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    people_visiting = st.number_input(
        "Number of People Visiting", min_value=1, max_value=10, value=2, step=1
    )
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5], index=0)
    marital_status = st.selectbox(
        "Marital Status", ["Married", "Divorced", "Unmarried", "Single"]
    )
    trips = st.number_input("Number of Trips per Year", min_value=0, max_value=30, value=3)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    children = st.number_input(
        "Number of Children Visiting (below 5)", min_value=0, max_value=10, value=0
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0, max_value=200000, value=25000, step=1000
    )
    submitted = st.form_submit_button("Predict Purchase Likelihood")

if submitted:
    customer = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": people_visiting,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    try:
        model = load_model()
        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])
        st.metric("Estimated purchase probability", f"{probability:.1%}")
        if prediction == 1:
            st.success("The model identifies this customer as a potential buyer.")
        else:
            st.info("The model does not currently classify this customer as a likely buyer.")
        st.caption("This is a model-based estimate, not a guarantee of customer behavior.")
    except FileNotFoundError as error:
        st.error(str(error))
