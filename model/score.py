import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Załaduj dane
df = pd.read_csv('../data/cs-training.csv')  # <-- podmień na właściwą ścieżkę
features = [
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfDependents',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberRealEstateLoansOrLines',
    'NumberOfOpenCreditLinesAndLoans',
    'RevolvingUtilizationOfUnsecuredLines',
    'age'
]
# Czyszczenie: usuń ekstremalne DebtRatio
df = df[df['DebtRatio'] <= 1]
df = df[df['MonthlyIncome'] > 0]


# Wybierz cechy i target
X = df[features]  # tylko DebtRatio
y = df['SeriousDlqin2yrs']  # target binarny: 0/1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predykcja prawdopodobieństw
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Wyniki
print("AUC:", roc_auc_score(y_test, y_pred_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

coefficients = pd.Series(model.coef_[0], index=X.columns)
coefficients = coefficients.sort_values(ascending=False)
print(coefficients)

# Policzenie współczynnika niezbalansowania
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

# XGBoost z dopasowaniem do niezbalansowanych danych
model = XGBClassifier(
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
print("------------- XGBOOST -----------")
model.fit(X_train, y_train)

# Predykcje
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Ocena
print("AUC:", roc_auc_score(y_test, y_pred_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib

joblib.dump(model, 'xgboost_model.pkl')