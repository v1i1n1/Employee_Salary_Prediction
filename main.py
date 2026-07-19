from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# ==========================

import joblib

# ==========================
# Step 1 : Import Libraries
# ==========================

import pandas as pd

# ======================
# Step 2 : Load Dataset
# ======================

df = pd.read_csv("data/employee_salary.csv")

# ============================
# Step 3 : understanding the dataset
# ============================

print("========== Complete Dataset ==========")
print(df)

print("\n========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Shape ==========")
print(df.shape)

print("\n========== Column Names ==========")
print(df.columns)

print("\n========== Data Types ==========")
print(df.dtypes)

print("\n========== Dataset Information ==========")
df.info()

print("\n========== Statistical Summary ==========")
print(df.describe())

# ===========================
# step 4 : Data Cleaning
# ===========================

print("\n========== Missing Values ==========")
print(df.isnull().sum())

print("\n========== Duplicate Records ==========")
print(df.duplicated().sum())

# ==========================
# Step 5 : Feature Engineering
# ==========================

df["Performance_Index"] = df["Experience"] * df["Skill_Score"]

df["Learning_Efficiency"] = df["Skill_Score"] / df["Experience"]

print("\n========== Dataset After Feature Engineering ==========")
print(df)

# ==========================
# Step 6 : Encoding
# ==========================

education_mapping = {
    "B.Tech": 0,
    "M.Tech": 1,
    "MCA": 2
}

df["Education"] = df["Education"].map(education_mapping)

print("\n========== Dataset After Encoding ==========")
print(df)

# ==========================
# Step 7 : Feature Selection
# ==========================

features = [
    "Education",
    "Experience",
    "Skill_Score",
    "Performance_Index",
    "Learning_Efficiency"
]

target = "Salary"


X = df[features]
y = df[target]

print("\n========== Features (X) ==========")
print(X)

print("\n========== Target (y) ==========")
print(y)

# ==========================

from sklearn.model_selection import train_test_split

# ==========================
# Step 8 : Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== X Train ==========")
print(X_train)

print("\n========== X Test ==========")
print(X_test)

print("\n========== y Train ==========")
print(y_train)

print("\n========== y Test ==========")
print(y_test)

# ===========================

from sklearn.linear_model import LinearRegression

# ==============================

# ==========================
# Step 9 : Model Training
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

print("\n========== Model Coefficients ==========")
print(model.coef_)

print("\n========== Model Intercept ==========")
print(model.intercept_)

print("\n========== Model Training Completed ==========")

# ==========================
# Step 10 : Prediction
# ==========================

predictions = model.predict(X_test)

print("\n========== Predicted Salary ==========")
print(predictions)

print("\n========== Actual vs Predicted ==========")

#====================================================

comparison = X_test.copy()

comparison["Actual Salary"] = y_test.values
comparison["Predicted Salary"] = predictions

print(comparison)

# ==========================
# Step 11 : Model Evaluation
# ==========================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mean_squared_error(y_test, predictions) ** 0.5

r2 = r2_score(y_test, predictions)

print("\n========== Model Evaluation ==========")

print(f"Mean Absolute Error (MAE): {mae:.2f}")

print(f"Mean Squared Error (MSE): {mse:.2f}")

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

print(f"R2 Score: {r2:.4f}")

# ==========================================

joblib.dump(model, "salary_prediction_model.pkl")