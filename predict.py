# predict.py
import joblib
import pandas as pd

def predict_loan_status(input_data):
    # Load model & encoders
    model = joblib.load("loan_model.pkl")
    encoders = joblib.load("label_encoders.pkl")

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical fields using saved encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Predict
    prediction = model.predict(df)[0]
    return "Approved" if prediction == 1 else "Rejected"

if __name__ == "__main__":
    sample_input = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 200,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban"
    }
    print("Prediction:", predict_loan_status(sample_input))
