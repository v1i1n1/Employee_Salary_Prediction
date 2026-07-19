from flask import Flask, render_template, request
import pandas as pd
import joblib

# ==========================================
# Create Flask App
# ==========================================

app = Flask(__name__)

# ==========================================
# Load Trained Model
# ==========================================

model = joblib.load("salary_prediction_model.pkl")

# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    education = int(request.form["education"])
    experience = float(request.form["experience"])
    skill_score = int(request.form["skill_score"])

    # Feature Engineering

    performance_index = experience * skill_score
    learning_efficiency = skill_score / experience

    # Create DataFrame

    new_employee = pd.DataFrame({

        "Education": [education],
        "Experience": [experience],
        "Skill_Score": [skill_score],
        "Performance_Index": [performance_index],
        "Learning_Efficiency": [learning_efficiency]

    })

    # Prediction

    prediction = model.predict(new_employee)

    predicted_salary = f"{prediction[0]:,.2f}"

    return render_template(
        "index.html",
        prediction=predicted_salary
    )

# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)