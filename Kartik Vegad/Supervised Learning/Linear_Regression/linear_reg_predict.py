import joblib
import pandas as pd

# IMPORTANT → this import is required for pickle to rebuild the model
from Linear_Regression import LinearRegressionModel

MODEL_SAVE_PATH = "Supervised Learning/Linear_Regression/linear_regression_model.pkl"


def get_valid_input(prompt, valid_options=None, cast_type=str):
    while True:
        value = input(prompt).strip().lower()

        if cast_type != str:
            try:
                value = cast_type(value)
            except:
                print("Invalid value. Try again.")
                continue

        if valid_options and value not in valid_options:
            print(f"Allowed values: {', '.join(valid_options)}")
            continue

        return value


def predict_from_user_input():
    print("\n" + "=" * 70)
    print("INSURANCE CHARGES PREDICTION SYSTEM")
    print("=" * 70)

    # Load model
    saved = joblib.load(MODEL_SAVE_PATH)
    model = saved['model']
    feature_means = saved['feature_means']
    feature_stds = saved['feature_stds']
    feature_names = saved['feature_names']

    # ================= USER INPUT =================
    age = get_valid_input("Enter age: ", cast_type=int)
    bmi = get_valid_input("Enter BMI: ", cast_type=float)
    children = get_valid_input("Enter number of children: ", cast_type=int)

    sex = get_valid_input("Enter sex (male/female): ", ["male", "female"])
    smoker = get_valid_input("Smoker (yes/no): ", ["yes", "no"])
    region = get_valid_input(
        "Region (northeast/northwest/southeast/southwest): ",
        ["northeast", "northwest", "southeast", "southwest"]
    )

    # ================= PREPARE DATA =================
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # One hot encode
    input_df = pd.get_dummies(input_df)

    # Add missing columns from training
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Maintain same order
    input_df = input_df[feature_names]

    # Standardise
    input_df = (input_df - feature_means) / feature_stds

    # ================= PREDICTION =================
    prediction = model.predict(input_df.values)[0]

    # Clip negative values
    prediction = max(0, prediction)

    print("\n" + "-" * 40)
    print(f"Estimated Insurance Charges: ₹ {prediction:,.2f}")
    print("-" * 40)


if __name__ == "__main__":
    predict_from_user_input()
