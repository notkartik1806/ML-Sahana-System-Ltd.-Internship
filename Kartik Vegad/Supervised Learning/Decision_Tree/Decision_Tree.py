# Decision_Tree.py

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from synthetic_generator import generate_synthetic_data

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

MAX_DEPTH_RANGE = range(1, 15)
CRITERION = "gini"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_model.pkl")

os.makedirs(GRAPH_DIR, exist_ok=True)

# =============================================================================
# METRICS
# =============================================================================

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


# =============================================================================
# PROCESSOR
# =============================================================================

class DatasetProcessor:
    def __init__(self):
        self.encoders = {}

    def fit_transform(self, X, y):
        X_encoded = X.copy()

        for col in X.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            self.encoders[col] = le

        y_le = LabelEncoder()
        y_encoded = y_le.fit_transform(y)
        self.encoders["target"] = y_le

        return X_encoded, y_encoded


# =============================================================================
# TUNER
# =============================================================================

class DecisionTreeTuner:
    def tune_depth(self, X_train, y_train, X_val, y_val):
        accuracies = []

        for depth in MAX_DEPTH_RANGE:
            model = DecisionTreeClassifier(
                max_depth=depth,
                criterion=CRITERION,
                random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)
            acc = accuracy_score(y_val, model.predict(X_val))
            accuracies.append(acc)

        best_depth = list(MAX_DEPTH_RANGE)[np.argmax(accuracies)]

        plt.figure(figsize=(10, 6))
        plt.plot(list(MAX_DEPTH_RANGE), accuracies, marker="o")
        plt.title("Accuracy vs Max Depth")
        plt.savefig(os.path.join(GRAPH_DIR, "accuracy_vs_depth.png"))
        plt.close()

        return best_depth


# =============================================================================
# PIPELINE
# =============================================================================

class MLPipeline:
    def run(self):

        print("Generating synthetic tennis dataset...")

        X, y = generate_synthetic_data(
            n_samples=1000,
            save=True
        )

        print("Synthetic dataset saved inside the data folder.")


        processor = DatasetProcessor()
        X, y = processor.fit_transform(X, y)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            stratify=y, random_state=RANDOM_STATE
        )

        val_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=RANDOM_STATE
        )

        tuner = DecisionTreeTuner()
        best_depth = tuner.tune_depth(X_train, y_train, X_val, y_val)

        model = DecisionTreeClassifier(
            max_depth=best_depth,
            criterion=CRITERION,
            random_state=RANDOM_STATE
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        metrics = ClassificationMetrics(
            accuracy=accuracy_score(y_test, preds),
            precision=precision_score(y_test, preds),
            recall=recall_score(y_test, preds),
            f1_score=f1_score(y_test, preds)
        )

        print("\nBest Depth:", best_depth)
        print("Accuracy :", round(metrics.accuracy, 4))
        print("Precision:", round(metrics.precision, 4))
        print("Recall   :", round(metrics.recall, 4))
        print("F1 Score :", round(metrics.f1_score, 4))

        joblib.dump(model, MODEL_PATH)
        print(f"\nModel saved at: {MODEL_PATH}")


def main():
    MLPipeline().run()


if __name__ == "__main__":
    main()
