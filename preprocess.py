# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    # Load CSV file
    df = pd.read_csv(filepath)

    # Example preprocessing: handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features & target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders = load_and_preprocess_data("loan_data.csv")
    print("Data preprocessing completed!")
