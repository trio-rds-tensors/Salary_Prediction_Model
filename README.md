# 💸 Salary Prediction Model

This repository contains a Machine Learning model that predicts Salary based on input features such as Education Level , Job Tittle and Year of Experience. The model has been trained and saved in `.py` and `pkl`format for reuse and deployment.

## 👥 Project Team
- Dipali Gantait
- Sudipta Singha
- Riyajul Saha

## 📁 Project Contents

- `README.md` – Overview and instructions.
- `salary_prediction_model.pkl` – Trained ML model saved using `pickle`.
- `Salary.csv` – Dataset.
- `Salary_Prediction_Model.ipynb` - Source Code.
- `Salary_Prediction_Model.py` – For Easily Reuse Model.
- `preprocess.py`- Preprocess data before use model
- `Salary_Prediction_Model_YSPM_Dipali-Sudipta-Riyajul.pdf` – Complete Report papaer of Model.

## 🧠 Model Information

- **Model Type**: Multilinear Regression
- **Purpose**: Predict Salary.
- **Input Features**:  Education Level , Job Tittle and Year of Experience
- **Output**: Predicted Salary

## 🧪 How to Use the Model
### 🕵️ Method 1

1. Clone this repository:
   ```bash
   git clone https://github.com/trio-rds-tensors/Salary_Prediction_Model.git

2. Install required libraries:
   ```bash
   pip install numpy pandas scikit-learn
3. Load and use the model:
   ```bash
   from preprocess import preprocessor
   import pickle
   import pandas as pd
   # Load model
   with open('salary_prediction_model.pkl','rb') as file:
      model = pickle.load(file)
   #Predict (example)
   predicted_salary=model.predict(pd.DataFrame([[Edu,Job,Exp]],columns=["Education Level","Job Title","Years of Experience"])
   print(predicted_salary)
### 🕵️ Method 2
1. Clone this repository:
   ```bash
   git clone https://github.com/trio-rds-tensors/Salary_Prediction_Model.git
   ```
2. Create and use model
   ```bash
   from Salary_Prediction_Model.py import Salary
```bash
   model = Salary()
   model.predict([Edu,Job,Exp])
```
## ✅ Example
   ```bash
   # Method 1
   model.predict([["PhD","Senior Scientist",11]])
```
  ```bash
#Method 2
  model.predict(["PhD","Senior Scientist",11])
```
## 📊 Performance
**Accuracy Metric:** R² Score = 0.8574516839044533

The model has been evaluated on test data and performs well on predicting continuous score values.

## 📜 License
This project is open-source and free to use for educational purposes.
