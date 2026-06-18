import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from data_cleaning import clean_loan_dataset



print("Loading and cleaning dataframe")
df = clean_loan_dataset("../data/loan_approval_data.csv")
print("Loading and cleaning successfull")
print(df.info())

print("Separating features and target")
x=df.drop("Loan_Approved",axis=1)
y=df["Loan_Approved"]
print("Separation successfull")


# SPLITTING
print("Splitting datas")
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)
print("Splitting completed")

# SCALLING 
print("Scalling the dataset")
scaler=StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
print("train scalling done")
x_test_scaled = scaler.transform(x_test)
print("test scalling done")



# TRAINNING THE GAUSSIAN NAIVE BAYES MODEL
print("Training Gaussian Naive Bayes Model")
gnb_model = GaussianNB()
gnb_model.fit(x_train_scaled,y_train)
print("Training Done")



# EXPORT THE TRAINED MODEL
print("Saving Files...")
joblib.dump(gnb_model, "../models/loan_approval_model.pkl")
print("    Model  Saved  ")
joblib.dump(scaler,"../models/main_scaler.pkl")
print("  Scaler  Model  Saved  ")
print("Pipeline executed perfectly! Model and Scaler saved. ")


