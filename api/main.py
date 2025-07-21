from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Załaduj model
model = joblib.load("../model/xgboost_model.pkl")

# API app
app = FastAPI(title="Credit Scoring API")


# Schemat wejściowy (dostosuj do swoich cech!)
class ClientData(BaseModel):
    DebtRatio: float
    MonthlyIncome: float
    NumberOfDependents: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    NumberOfTimes90DaysLate: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberRealEstateLoansOrLines: int
    NumberOfOpenCreditLinesAndLoans: int
    RevolvingUtilizationOfUnsecuredLines: float
    age: int


@app.post("/predict")
def predict(data: ClientData):
    features = np.array([[
        data.DebtRatio,
        data.MonthlyIncome,
        data.NumberOfDependents,
        data.NumberOfTime30_59DaysPastDueNotWorse,
        data.NumberOfTimes90DaysLate,
        data.NumberOfTime60_89DaysPastDueNotWorse,
        data.NumberRealEstateLoansOrLines,
        data.NumberOfOpenCreditLinesAndLoans,
        data.RevolvingUtilizationOfUnsecuredLines,
        data.age
    ]])

    prob = model.predict_proba(features)[0][1]
    label = int(prob >= 0.5)

    return {
        "default_probability": round(prob, 4),
        "prediction": label
    }