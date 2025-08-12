# train_model.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data

def train_and_save_model():
    X_train, X_test, y_train, y_test, encoders = load_and_preprocess_data("loan_data.csv")

    # Train Logistic Regression
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Test accuracy
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Save model & encoders
    joblib.dump(model, "loan_model.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    print("Model and encoders saved!")

if __name__ == "__main__":
    train_and_save_model()
