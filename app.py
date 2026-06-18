import streamlit as st
# import joblib 

from src.data_cleaning import clean_loan_dataset_client


model = joblib.load("models/loan_approval_model.pkl")
scaler = joblib.load("models/main_scaler.pkl")


st.set_page_config(page_title="Credit risk assessment system", layout="wide")
st.title(" BANK LOAN APPROVAL SYSTEM")
st.write("Input applicant matrics below to calculate credit risk assessment. ")
st.markdown("---")

col1, col2, col3= st.columns(3)

with col1:
    st.subheader("Identity & Demographics")
    applicant_id=st.text_input("Applicant ID",value="LP001001",help="Unique identifier (Excluded from ML model calculation )")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender",["Male","Female"])
    marital_status = st.selectbox("Marital Status ", ["Single" , "Married"])
    dependents=st.selectbox("Dependents ",["0","1","2","3"])

with col2:
    st.subheader("Financial Profile")
    applicant_income = st.number_input("Applicant Monthly Income ($) ", min_value=0, value=0)
    coapplicant_income = st.number_input("Co - Applicant Monthly Income ($) ", min_value=0, value=0)
    saving = st.number_input("Total Savings Balance ($) ",min_value=0, value=15000)
    dti_ratio = st.slider("Debt - to - Income ",min_value=0.0, max_value=1.0, step=0.001, value=0.3, help="Total monthly debt payments devided by gross monthly income" )
    credit_score = st.slider("Credit Score ",min_value=300, max_value=850, value=650)

with col3:
    st.subheader("Employment and loan details ")
    employment_status = st.selectbox("Employment Status ",["Salaried","Self-employed","Contract","Unemployed"])
    employer_category = st.selectbox("Employment Category ",["Private","Government","MNC","Business","Unemployed"])
    edu_level = st.selectbox("Educational level ",["Graduate","Not Graduate"])
    existing_loans = st.number_input("Number of existing loans",min_value=0, value=0, step=1)
    property_area = st.selectbox("Property Area ",["Urban","Semiurban","Rural"])

    st.markdown("---")
    st.subheader("Requested loan parameter")
    col4, col5, col6 = st.columns(3)
    with col4:
        loan_amount=st.number_input("Requested Loan Amount ($)",min_value=0, value=25000)
    with col5:
        loan_term=st.number_input("Loan Term (In Months)",min_value=1, max_value=360, value=36, step=1)
    with col6:
        loan_purpose=st.selectbox("Loan Purposse",["Personal","Car","Business","Home","Education"])
        collateral_value = st.number_input("Collateral Value ($) ",min_value=0, value=0)
st.markdown("###") # Add spacing



if st.button(" Calculate Credit Risk Assessment ", type="primary", use_container_width=True):

    df = clean_loan_dataset_client(age, gender, marital_status, dependents, applicant_income, coapplicant_income, saving, dti_ratio, credit_score, employment_status, employer_category, edu_level, existing_loans, property_area, loan_amount, loan_term, loan_purpose, collateral_value)


    scaled_features=scaler.transform(df)

    print(scaled_features)

    prediction = model.predict(scaled_features)

    probability = model.predict_proba(scaled_features)[0][1]

    st.subheader("--- Analysis Result ----")
    if prediction[0]==1:
        st.success(f"Loan Approved ! (Confidence:{probability*100:.1f}%)")
    else:
        st.error(f"Loan Rejected: High Default risk profile. (Risk Probability: {(1-probability)*100:.1f}%)")


