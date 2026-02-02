from typing import Dict, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


# 1. Data_Loading
class Data_Loading:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)


# 2. Data_Verifying
class Data_Verifying:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def verify(self) -> None:
        print("Shape of the Dataset:", self.df.shape)
        print("Columns in the Dataset:", self.df.columns.tolist())
        print("Data Types of Columns:\n", self.df.dtypes)

        print("\nFirst 5 Rows of Dataset:\n", self.df.head())
        print("\nLast 5 Rows of Dataset:\n", self.df.tail())
        print("\nRandom Sample from Dataset:\n", self.df.sample(5))

        print("\nStatistical Description of Dataset:\n", self.df.describe())

        print("\nMissing Values in Each Column:\n", self.df.isnull().sum())
        print("Number of Duplicate Rows:", self.df.duplicated().sum())

        print("\nUnique Values in Categorical Columns:")
        for col in self.df.select_dtypes(include="object").columns:
            print(f"{col}:", self.df[col].unique())

        print("\nValue Counts of Categorical Columns:")
        for col in self.df.select_dtypes(include="object").columns:
            print(f"\n{col}:\n", self.df[col].value_counts())

        print("\nCorrelation Matrix:\n", self.df.corr(numeric_only=True))


# 3. Data_Validation
class Data_Validation:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def validate(self) -> Dict[str, bool]:
        return {
            "Missing Values Present": self.df.isnull().any().any(),
            "Duplicate Rows Present": self.df.duplicated().any(),
            "Invalid Age Values": ((self.df["age"] <= 0) | (self.df["age"] > 120)).any(),
            "Negative Charges Present": (self.df["charges"] < 0).any(),
            "Invalid Smoker Values":
                not set(self.df["smoker"].unique()).issubset({"yes", "no"})
        }


# 4. Data_Processor
class Data_Processor:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def process(self) -> pd.DataFrame:
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        return self.df


# 5. Data_Visualisation
class Data_Visualisation:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def visualise(self) -> None:
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            if col != "charges":
                sns.scatterplot(x=self.df[col], y=self.df["charges"])
                plt.title(f"{col} vs Charges")
                plt.show()

        for col in ["sex", "smoker", "region"]:
            sns.boxplot(x=self.df[col], y=self.df["charges"])
            plt.title(f"{col} vs Charges")
            plt.show()


# 6. Dataset_Splitting
class Dataset_Splitting:
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               np.ndarray, np.ndarray, np.ndarray]:

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values


# 7. Dataset_Transforming
class Dataset_Transforming:
    def __init__(self):
        self.transformer = None

    def fit(self, X_train: pd.DataFrame) -> None:
        numeric_features = ["age", "bmi", "children"]
        categorical_features = ["sex", "smoker", "region"]

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first"), categorical_features)
            ]
        )

        self.transformer.fit(X_train)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.transformer.transform(X)


# 8. LinearRegression (CUSTOM)
class LinearRegression:
    def __init__(self):
        self.model = SklearnLinearRegression()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 9. Model_Training
class Model_Training:
    def __init__(self, model: LinearRegression):
        self.model = model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.train(X, y)


# 10. Model_Evaluation
class Model_Evaluation:
    def evaluate(self, model: LinearRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        predictions = model.predict(X)
        return {
            "MAE": mean_absolute_error(y, predictions),
            "MSE": mean_squared_error(y, predictions),
            "RMSE": np.sqrt(mean_squared_error(y, predictions)),
            "R2": r2_score(y, predictions)
        }



# 11. Pipeline
class Pipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def run(self) -> None:
        df = Data_Loading(self.data_path).load()

        Data_Verifying(df).verify()

        validation_report = Data_Validation(df).validate()
        print("\nValidation Report:", validation_report)

        df_clean = Data_Processor(df).process()

        Data_Visualisation(df_clean).visualise()

        X = df_clean.drop("charges", axis=1)
        y = df_clean["charges"]

        splitter = Dataset_Splitting()
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = splitter.split(X, y)

        transformer = Dataset_Transforming()
        transformer.fit(X_train_raw)

        X_train = transformer.transform(X_train_raw)
        X_test = transformer.transform(X_test_raw)

        model = LinearRegression()
        Model_Training(model).train(X_train, y_train)

        metrics = Model_Evaluation().evaluate(model, X_test, y_test)
        print("\nModel Evaluation Metrics:", metrics)


if __name__ == "__main__":
    Pipeline("Supervised Learning\Linear Regression\medical_insurance.csv").run()

