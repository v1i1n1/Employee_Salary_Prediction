# ==========================================
# Employee Salary Prediction - Prediction File
# ==========================================

import joblib
import pandas as pd

# ==========================================
# Load Saved Model
# ==========================================

model = joblib.load("salary_prediction_model.pkl")

print("=" * 50)
print("     EMPLOYEE SALARY PREDICTION SYSTEM")
print("=" * 50)

# ==========================================
# Education Mapping
# ==========================================

education_mapping = {
    "b.tech": 0,
    "m.tech": 1,
    "mca": 2
}

display_name = {
    "b.tech": "B.Tech",
    "m.tech": "M.Tech",
    "mca": "MCA"
}

# ==========================================
# Get User Input
# ==========================================

education_text = input("\nEnter Education (B.Tech / M.Tech / MCA): ").strip().lower()

if education_text not in education_mapping:
    print("\nInvalid Education!")
    print("Please enter only: B.Tech, M.Tech or MCA")
    exit()

education = education_mapping[education_text]

experience = float(input("Enter Experience (Example: 5.5): "))
if experience <= 0:
    print("Experience must be greater than 0.")
    exit()

skill_score = int(input("Enter Skill Score (0-100): "))
if skill_score < 0 or skill_score > 100:
    print("Skill Score must be between 0 and 100.")
    exit()

# ==========================================
# Feature Engineering
# ==========================================

performance_index = experience * skill_score
learning_efficiency = skill_score / experience

# ==========================================
# Prepare Data for Prediction
# ==========================================

new_employee = pd.DataFrame({
    "Education": [education],
    "Experience": [experience],
    "Skill_Score": [skill_score],
    "Performance_Index": [performance_index],
    "Learning_Efficiency": [learning_efficiency]
})

# ==========================================
# Predict Salary
# ==========================================

predicted_salary = model.predict(new_employee)

# ==========================================
# Display Results
# ==========================================

print("\n" + "=" * 50)
print("Prediction Result")
print("=" * 50)

print(f"Education           : {display_name[education_text]}")
print(f"Experience          : {experience:.1f} Years")
print(f"Skill Score         : {skill_score}")
print(f"Performance Index   : {performance_index:.2f}")
print(f"Learning Efficiency : {learning_efficiency:.2f}")

print("\nPredicted Salary")
print(f"₹ {predicted_salary[0]:,.2f}")

print("=" * 50)
print("Prediction Completed Successfully")
print("=" * 50)