# 💳 Online Payments Fraud Detection System

A Machine Learning-based web application that detects fraudulent online payment transactions using a trained Random Forest model and an interactive Streamlit interface.

---

## 📌 Project Overview

Online payment systems are vulnerable to fraudulent activities. This project uses supervised machine learning to classify transactions as **Fraudulent** or **Legitimate** based on transaction behavior patterns.

The system:
- Trains a Random Forest model on transaction data
- Predicts fraud probability
- Classifies risk level (Low / Medium / High)
- Provides a user-friendly Streamlit web interface

---

## 🧠 Machine Learning Model

**Algorithm Used:** Random Forest Classifier  

**Why Random Forest?**
- Handles high-dimensional data effectively
- Reduces overfitting via ensemble learning
- Performs well on imbalanced datasets
- Provides probability-based predictions

**Fraud Detection Strategy:**
- Uses transaction features such as balance changes and transaction type
- Calculates fraud probability
- Applies threshold-based classification

---

## 📊 Features Used for Prediction

The model uses the following input features:

- `step`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `type`

---

## ⚙️ Risk Classification Logic

Based on fraud probability:

- **Probability < 0.5 → Low Risk**
- **0.5 ≤ Probability < 0.8 → Medium Risk**
- **Probability ≥ 0.8 → High Risk**

---

## 🖥️ Streamlit Web Interface

The web application allows users to:

- Enter transaction details manually
- Select transaction type
- Get fraud probability instantly
- View risk classification
- Detect fraudulent transactions in real-time

---

## 📂 Project Structure

```
online-payments-fraud-detection-using-ml/
│
├── model/
│   ├── fraud_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── fraud_model_training.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/online-payments-fraud-detection.git
cd online-payments-fraud-detection
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser automatically.

---

## 🧪 Example Fraud Test Case

Try this input:

```
Step: 1
Amount: 100000
Old Balance Origin: 100000
New Balance Origin: 0
Old Balance Destination: 0
New Balance Destination: 100000
Transaction Type: TRANSFER
```

Expected Result:
- High Fraud Probability
- High Risk

---

## 📈 Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

For fraud detection, **Recall and ROC-AUC** are prioritized over accuracy.

---

## 🔐 Important Notes

- The dataset file (`fraud.csv`) is excluded from GitHub via `.gitignore`.
- The model must be retrained if feature structure changes.
- Transaction type encoding must match training encoding.

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib

---

## 🎯 Future Improvements

- Threshold tuning slider
- Confusion matrix dashboard
- CSV bulk upload prediction
- Deployment to Streamlit Cloud
- Model optimization using XGBoost

---

## 📜 License

This project is for educational and academic purposes.

---

## 👩‍💻 Author

**Kavyamrutha Puvvuthota**

Machine Learning | Data Science | Web App Development

---

⭐ If you found this project useful, consider giving it a star!
