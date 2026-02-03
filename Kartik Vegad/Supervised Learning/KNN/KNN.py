from typing import Dict, Tuple
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 1. Data_Loading
class Data_Loading:
    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        iris = load_iris()

        X = pd.DataFrame(
            iris.data,
            columns=iris.feature_names
        )

        y = pd.Series(
            iris.target,
            name="species"
        )

        return X, y


# 2. Data_Verifying
class Data_Verifying:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def verify(self) -> None:
        print("Shape of Feature Dataset:", self.X.shape)
        print("Feature Columns:", self.X.columns.tolist())
        print("Feature Data Types:\n", self.X.dtypes)

        print("\nTarget Distribution:\n", self.y.value_counts())

        print("\nFirst 5 Rows of Features:\n", self.X.head())


# 3. Data_Validation
class Data_Validation:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def validate(self) -> Dict[str, bool]:
        return {
            "Missing Values in Features": self.X.isnull().any().any(),
            "Missing Values in Target": self.y.isnull().any(),
            "Duplicate Feature Rows": self.X.duplicated().any(),
            "Invalid Target Labels":
                not set(self.y.unique()).issubset({0, 1, 2})
        }


# 4. Data_Processor
class Data_Processor:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.copy()
        self.y = y.copy()

    def process(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y



# 5. Dataset_Splitting
class Dataset_Splitting:
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               np.ndarray, np.ndarray, np.ndarray]:

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values


# 6. Dataset_Transforming
class Dataset_Transforming:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame) -> None:
        self.scaler.fit(X_train)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X)


# 7. KNN_Model
class KNN_Model:
    def __init__(self, k: int = 5):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# 8. Model_Training
class Model_Training:
    def __init__(self, model: KNN_Model):
        self.model = model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.train(X, y)


# 9. Model_Evaluation
class Model_Evaluation:
    def evaluate(
        self,
        model: KNN_Model,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:

        predictions = model.predict(X)

        return {
            "Accuracy": accuracy_score(y, predictions),
            "Precision": precision_score(y, predictions, average="macro"),
            "Recall": recall_score(y, predictions, average="macro"),
            "F1 Score": f1_score(y, predictions, average="macro")
        }

    def confusion_matrix(
        self,
        model: KNN_Model,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:

        return confusion_matrix(y, model.predict(X))


# 10. Pipeline
class Pipeline:
    def __init__(self, k: int = 5):
        self.k = k

    def run(self) -> None:
        X, y = Data_Loading().load()

        Data_Verifying(X, y).verify()

        validation_report = Data_Validation(X, y).validate()
        print("\nValidation Report:", validation_report)

        X_clean, y_clean = Data_Processor(X, y).process()

        splitter = Dataset_Splitting()
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = splitter.split(
            X_clean, y_clean
        )

        transformer = Dataset_Transforming()
        transformer.fit(X_train_raw)

        X_train = transformer.transform(X_train_raw)
        X_test = transformer.transform(X_test_raw)

        model = KNN_Model(k=self.k)
        Model_Training(model).train(X_train, y_train)

        metrics = Model_Evaluation().evaluate(model, X_test, y_test)
        print("\nModel Evaluation Metrics:", metrics)

        print(
            "\nConfusion Matrix:\n",
            Model_Evaluation().confusion_matrix(model, X_test, y_test)
        )


if __name__ == "__main__":
    Pipeline(k=5).run()
