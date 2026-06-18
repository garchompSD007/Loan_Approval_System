import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.impute import SimpleImputer

# def clean_loan_dataset(filepath):
#     df=pd.read_csv(filepath)
#     df=df.drop("Applicant_ID",axis=1)

#     categorical_cols = df.select_dtypes(include=["object"]).columns 
#     numerical_cols = df.select_dtypes(include=["float64","int64"]).columns 

#     num_imp = SimpleImputer(strategy="mean")
#     df[numerical_cols]=num_imp.fit_transform(df[numerical_cols])

#     cat_imp = SimpleImputer(strategy="most_frequent")
#     df[categorical_cols]=cat_imp.fit_transform(df[categorical_cols])

#     le = LabelEncoder()
#     df["Education_Level"]=le.fit_transform(df["Education_Level"])
#     df["Loan_Approved"]=le.fit_transform(df["Loan_Approved"])

#     labelled_cols=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
#     ohe=OneHotEncoder(drop="first" , sparse_output=False, handle_unknown="ignore")
#     encoded=ohe.fit_transform(df[labelled_cols])
#     encoded_df=pd.DataFrame(encoded, columns=ohe.get_feature_names_out(labelled_cols), index=df.index)
#     df=pd.concat([df.drop(columns=labelled_cols), encoded_df],axis=1)

#     # FEATURE ENGINEERING add or transforming feature
#     df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
#     df["Credit_Score_sq"] = df["Credit_Score"]**2
#     df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

#     return df

# taking dataframe from user

def clean_loan_dataset_client(age, gender, marital_status, dependents, applicant_income, coapplicant_income, saving, dti_ratio, credit_score, employment_status, employer_category, edu_level, existing_loans, property_area, loan_amount, loan_term, loan_purpose, collateral_value):

    # label encodding and one hot encodding
    # "Education_Level"
    if edu_level == "Graduate":
        edu_level = 0
    else:
        edu_level = 1

    # Employment status
    emp_st_sa = 0
    emp_st_se = 0
    emp_st_u = 0
    if employment_status == "Salaried":
        emp_st_sa = 1
        emp_st_se = 0
        emp_st_u = 0
    elif employment_status == "Self-employed":
        emp_st_sa = 0
        emp_st_se = 1
        emp_st_u = 0
    elif employment_status == "Unemployed":
        emp_st_sa = 0
        emp_st_se = 0
        emp_st_u = 1
    else:
        emp_st_sa = 0
        emp_st_se = 0
        emp_st_u = 0

    # Marital Status
    mar_st_s = 0
    if marital_status == "Single":
        mar_st_s = 1
    else:
        mar_st_s = 0


    # loan purpose

    loan_purpose_car = 0
    loan_purpose_education = 0
    loan_purpose_home = 0
    loan_purpose_personal = 0

    if loan_purpose == "Persoal":
        loan_purpose_car = 0
        loan_purpose_education = 0
        loan_purpose_home = 0
        loan_purpose_personal = 1
    elif loan_purpose == "Car":
        loan_purpose_car = 1
        loan_purpose_education = 0
        loan_purpose_home = 0
        loan_purpose_personal = 0
    elif loan_purpose == "Education":
        loan_purpose_car = 0
        loan_purpose_education = 1
        loan_purpose_home = 0
        loan_purpose_personal = 0
    elif loan_purpose == "Home":
        loan_purpose_car = 0
        loan_purpose_education = 0
        loan_purpose_home = 1
        loan_purpose_personal = 0
    else:
        loan_purpose_car = 0
        loan_purpose_education = 0
        loan_purpose_home = 0
        loan_purpose_personal = 0

    # Property area
    property_area_semiurban = 0
    property_area_urban = 0

    if property_area == "Semiurban":
        property_area_semiurban = 1
        property_area_urban = 0
    elif property_area == "Urban":
        property_area_semiurban = 0
        property_area_urban = 1
    else:
        property_area_semiurban = 0
        property_area_urban = 0

    # Gender
    gender_male = 0
    if gender == "Male":
        gender_male = 1
    else:
        gender_male = 0

    # Employment category

    employer_category_govt = 0
    employer_category_mnc = 0
    employer_category_pvt = 0
    employer_category_u = 0

    if employer_category == "Government":
        employer_category_govt = 1
        employer_category_mnc = 0
        employer_category_pvt = 0
        employer_category_u = 0

    elif employer_category == "MNC":
        employer_category_govt = 0
        employer_category_mnc = 1
        employer_category_pvt = 0
        employer_category_u = 0

    elif employer_category == "Private":
        employer_category_govt = 0
        employer_category_mnc = 0
        employer_category_pvt = 1
        employer_category_u = 0
    elif employer_category == "Unemployed":
        employer_category_govt = 0
        employer_category_mnc = 0
        employer_category_pvt = 0
        employer_category_u = 1

    else:
        employer_category_govt = 0
        employer_category_mnc = 0
        employer_category_pvt = 0
        employer_category_u = 0
    df = pd.DataFrame(
        {
            # "Applicant_ID":[applicant_id],
            "Applicant_Income":[applicant_income],
            "Coapplicant_Income":[coapplicant_income],
            "Age":[age],
            "Dependents":[dependents],
            "Credit_Score":[credit_score],
            "Existing_Loans":[existing_loans],
            "DTI_Ratio":[dti_ratio],
            "Savings":[saving],
            "Collateral_Value":[collateral_value],
            "Loan_Amount":[loan_amount],
            "Loan_Term":[loan_term],
            "Education_Level":[edu_level],
            "Employment_Status_Salaried":[emp_st_sa],
            "Employment_Status_Self-employed":[emp_st_se],
            "Employment_Status_Unemployed":[emp_st_u],
            "Marital_Status_Single":[mar_st_s],
            "Loan_Purpose_Car":[loan_purpose_car],
            "Loan_Purpose_Education":[loan_purpose_education],
            "Loan_Purpose_Home":[loan_purpose_home],
            "Loan_Purpose_Personal":[loan_purpose_personal],
            "Property_Area_Semiurban":[property_area_semiurban],
            "Property_Area_Urban":[property_area_urban],
            "Gender_Male":[gender_male],
            "Employer_Category_Government":[employer_category_govt],
            "Employer_Category_MNC":[employer_category_mnc],
            "Employer_Category_Private":[employer_category_pvt],
            "Employer_Category_Unemployed":[employer_category_u],
            "DTI_Ratio_sq":[dti_ratio**2],
            "Credit_Score_sq":[credit_score**2],
            "Applicant_Income_log":[np.log1p(applicant_income)]
        }
    )

    return df

