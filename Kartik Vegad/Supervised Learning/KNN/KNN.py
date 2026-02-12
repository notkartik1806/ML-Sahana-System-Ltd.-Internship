import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

DATASET_NAME = "Iris Dataset"
RANDOM_STATE = 42

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

K_NEIGHBORS = 5

MODEL_SAVE_PATH = "Supervised Learning/KNN/knn_model.pkl"

FIGURE_SIZE = (12, 8)
DPI = 100
STYLE = "seaborn-v0_8-darkgrid"


# =============================================================================
# METRICS DATA CLASS
# =============================================================================

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def __str__(self) -> str:
        return (
            f"\nModel Performance Metrics\n"
            f"{'=' * 50}\n"
            f"Accuracy  : {self.accuracy:.4f}\n"
            f"Precision : {self.precision:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"F1 Score  : {self.f1_score:.4f}\n"
            f"{'=' * 50}"
        )


# =============================================================================
# DATASET LOADER
# =============================================================================

class DatasetLoader:
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="species")

        print("✓ Dataset loaded successfully")
        print(f"✓ Samples: {X.shape[0]}")
        print(f"✓ Features: {X.shape[1]}")
        print(f"✓ Target classes: {y.unique()}")

        return X, y


# =============================================================================
# DATASET VALIDATOR
# =============================================================================

class DatasetValidator:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def verify(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        if self.X.empty or self.y.empty:
            print("✗ ERROR: Dataset is empty")
            return False

        print("✓ Dataset is not empty")
        print(f"✓ Feature shape: {self.X.shape}")
        print(f"✓ Target distribution:\n{self.y.value_counts()}")

        if self.X.isnull().any().any():
            print("⚠ WARNING: Missing values detected in features")
        else:
            print("✓ No missing values in features")

        if self.y.isnull().any():
            print("⚠ WARNING: Missing values detected in target")
        else:
            print("✓ No missing values in target")

        return True


# =============================================================================
# DATASET PROCESSOR
# =============================================================================

class DatasetProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X_train)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X)


# =============================================================================
# DATASET VISUALIZER
# =============================================================================

class DatasetVisualizer:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        plt.style.use(STYLE)

    def visualize(self) -> None:
        print(f"\n{'=' * 70}")
        print("DATASET VISUALIZATION")
        print(f"{'=' * 70}")

        sns.pairplot(
            pd.concat([self.X, self.y], axis=1),
            hue="species",
            corner=True
        )

        plt.savefig(
            "Supervised Learning/KNN/graphs/pairplot.png",
            dpi=DPI
        )
        plt.close()

        print("✓ Pairplot saved")


# =============================================================================
# KNN MODEL
# =============================================================================

class KNNModel:
    def __init__(self, k: int = K_NEIGHBORS):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:
    def __init__(self, model: KNNModel):
        self.model = model

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        dataset_name: str
    ) -> ClassificationMetrics:

        print(f"\n{'=' * 70}")
        print(f"MODEL EVALUATION - {dataset_name}")
        print(f"{'=' * 70}")

        y_pred = self.model.predict(X)

        metrics = ClassificationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average="macro"),
            recall=recall_score(y_true, y_pred, average="macro"),
            f1_score=f1_score(y_true, y_pred, average="macro")
        )

        print(metrics)

        self._plot_confusion_matrix(
            confusion_matrix(y_true, y_pred),
            dataset_name
        )

        return metrics

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        dataset_name: str
    ) -> None:

        plt.figure(figsize=(6, 5), dpi=DPI)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {dataset_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig(
            f"Supervised Learning/KNN/graphs/confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png",
            dpi=DPI
        )
        plt.close()

        print("✓ Confusion matrix saved")


# =============================================================================
# ML PIPELINE
# =============================================================================

class MLPipeline:
    def run(self) -> None:
        print(f"\n{'=' * 70}")
        print("KNN CLASSIFICATION PIPELINE")
        print(f"{'=' * 70}")

        # Load
        loader = DatasetLoader()
        X, y = loader.load_data()

        # Validate
        DatasetValidator(X, y).verify()

        # Visualize
        DatasetVisualizer(X, y).visualize()

        # Split dataset
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        val_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )

        print("\nDataset Split Summary:")
        print(f"Training samples:   {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples:       {X_test.shape[0]}")

        # Process
        processor = DatasetProcessor()
        X_train_scaled = processor.fit_transform(X_train)
        X_val_scaled = processor.transform(X_val)
        X_test_scaled = processor.transform(X_test)

        # Train model
        model = KNNModel()
        model.fit(X_train_scaled, y_train.values)

        # Evaluate
        evaluator = ModelEvaluator(model)
        evaluator.evaluate(X_train_scaled, y_train.values, "Training Set")
        evaluator.evaluate(X_val_scaled, y_val.values, "Validation Set")
        evaluator.evaluate(X_test_scaled, y_test.values, "Test Set")

        # Save model
        joblib.dump(
            {
                "model": model,
                "scaler": processor
            },
            MODEL_SAVE_PATH
        )

        print(f"\n✓ Model saved successfully at: {MODEL_SAVE_PATH}")
        print(f"\n{'=' * 70}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'=' * 70}")

def main():
    pipeline = MLPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
