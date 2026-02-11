import joblib
import pandas as pd
import numpy as np

# IMPORTANT → required so pickle can rebuild the model
from Logistic_Regression import LogisticRegressionModel


MODEL_SAVE_PATH = "Supervised Learning/Logistic_Regression/logistic_regression_model.pkl"


def get_valid_input(prompt, cast_type=str, valid_options=None):
    while True:
        value = input(prompt).strip()

        if cast_type != str:
            try:
                value = cast_type(value)
            except:
                print("Invalid value. Try again.")
                continue

        if valid_options and value.lower() not in valid_options:
            print(f"Allowed values: {', '.join(valid_options)}")
            continue

        return value


def predict_from_user_input():
    print("\n" + "=" * 70)
    print("SOCIAL NETWORK ADS PURCHASE PREDICTION")
    print("=" * 70)

    # Load model
    saved = joblib.load(MODEL_SAVE_PATH)
    model = saved["model"]
    feature_means = saved["feature_means"]
    feature_stds = saved["feature_stds"]
    feature_names = saved["feature_names"]
    encoders = saved.get("encoders", {})

    # ================= USER INPUT =================
    gender = get_valid_input("Gender (Male/Female): ", str)
    gender = gender.strip().lower().capitalize()

    age = get_valid_input("Age: ", int)
    salary = get_valid_input("Estimated Salary: ", float)

    # ================= CREATE DATAFRAME =================
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "EstimatedSalary": salary
    }])

    # ================= ENCODE =================
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # ================= ORDER =================
    input_df = input_df[feature_names]

    # ================= STANDARDISE =================
    input_df = (input_df - feature_means) / feature_stds

    # ================= PREDICTION =================
    probability = model.predict_proba(input_df.values)[0]
    prediction = model.predict(input_df.values)[0]

    label = "Purchased" if prediction == 1 else "Not Purchased"

    print("\n" + "-" * 50)
    print(f"Purchase Probability : {probability:.4f}")
    print(f"Model Prediction     : {label}")
    print("-" * 50)


if __name__ == "__main__":
    predict_from_user_input()
