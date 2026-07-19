# 💰 Employee Salary Prediction using Machine Learning

## 📌 Project Overview

Employee Salary Prediction is a Machine Learning Regression project developed using Python and Scikit-learn.

The objective of this project is to predict an employee's salary based on various factors such as:

- Education
- Experience
- Skill Score

The project also performs Feature Engineering by creating additional features to improve the model's prediction accuracy.

The trained model is saved using Joblib and can later be used to predict salaries for new employees without retraining the model.

---

# 🎯 Project Objective

The goal of this project is to understand the complete Machine Learning workflow including:

- Data Loading
- Data Analysis
- Data Cleaning
- Feature Engineering
- Encoding
- Feature Selection
- Model Training
- Prediction
- Model Evaluation
- Model Saving
- Model Reuse

---

# 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib

---

# 📂 Project Structure

```
Employee_Salary_Prediction/
│
├── data/
│   └── employee_salary.csv
│
├── main.py
├── predict.py
├── salary_prediction_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── venv/
```

---

# 📊 Dataset Information

The dataset contains employee information including:

| Column | Description |
|---------|-------------|
| Employee_ID | Unique Employee ID |
| Employee_Name | Employee Name |
| Education | Educational Qualification |
| Experience | Years of Experience |
| Skill_Score | Technical Skill Score |
| Salary | Employee Salary |

---

# ⚙️ Feature Engineering

Two additional features were created.

### Performance Index

```
Performance_Index = Experience × Skill_Score
```

### Learning Efficiency

```
Learning_Efficiency = Skill_Score / Experience
```

These engineered features help the model learn better relationships within the dataset.

---

# 🔄 Workflow

```
Load Dataset
      │
      ▼
Understand Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Encoding
      │
      ▼
Feature Selection
      │
      ▼
Train-Test Split
      │
      ▼
Linear Regression Model
      │
      ▼
Prediction
      │
      ▼
Model Evaluation
      │
      ▼
Save Model (.pkl)
```

---

# 🤖 Machine Learning Algorithm

The project uses

## Linear Regression

Linear Regression predicts the salary by learning the relationship between the independent variables and the target variable.

---

# 📈 Evaluation Metrics

The following metrics are used to evaluate the model.

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 💾 Model Saving

The trained model is saved using Joblib.

```python
joblib.dump(model, "salary_prediction_model.pkl")
```

The saved model can later be loaded without training again.

```python
model = joblib.load("salary_prediction_model.pkl")
```

---

# ▶️ How to Run the Project

## Step 1

Clone the repository.

```bash
git clone https://github.com/your-username/Employee_Salary_Prediction.git
```

---

## Step 2

Navigate to the project folder.

```bash
cd Employee_Salary_Prediction
```

---

## Step 3

Create a virtual environment.

```bash
python -m venv venv
```

---

## Step 4

Activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Step 5

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Step 6

Train the model.

```bash
python main.py
```

---

## Step 7

Predict salary.

```bash
python predict.py
```

---

# 📌 Sample Prediction

```
Enter Education (B.Tech / M.Tech / MCA): MCA

Enter Experience: 5

Enter Skill Score: 90

Predicted Salary

₹ 745,820.50
```

---

# 🚀 Future Improvements

- Flask Web Application
- Render Deployment
- User Authentication
- Database Integration
- Better Dataset
- Hyperparameter Tuning
- Random Forest Regression
- Decision Tree Regression
- XGBoost Regression
- Feature Importance Visualization
- Interactive Dashboard

---

# 📚 Learning Outcomes

Through this project, I learned:

- Python Programming
- Data Analysis using Pandas
- Feature Engineering
- Data Encoding
- Machine Learning Workflow
- Linear Regression
- Model Evaluation
- Model Serialization using Joblib
- Prediction using Saved Models
- Git & GitHub Project Management

---

# 👨‍💻 Author

**Vinodh Raj**

Machine Learning Enthusiast | Python Developer | ServiceNow Developer

---

# ⭐ If you found this project useful, please give it a Star.