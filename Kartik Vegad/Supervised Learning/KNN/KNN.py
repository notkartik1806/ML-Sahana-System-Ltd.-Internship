import warnings
from dataclasses import dataclass
from typing import Tuple

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
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

DATASET_NAME = "Iris Dataset"
RANDOM_STATE = 42

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# ---- HYPERPARAMETERS ----
K_RANGE = range(1, 31, 2)
DISTANCE_METRIC = "minkowski"
WEIGHTS = "distance"

MODEL_SAVE_PATH = "Supervised Learning/KNN/knn_model.pkl"

FIGURE_SIZE = (12, 8)
DPI = 100
STYLE = "seaborn-v0_8-darkgrid"

GRAPH_DIR = "Supervised Learning/KNN/graphs/"


# =============================================================================
# METRICS DATA CLASS
# =============================================================================

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


# =============================================================================
# DATASET LOADER
# =============================================================================

class DatasetLoader:
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="species")
        return X, y


# =============================================================================
# DATASET PROCESSOR
# =============================================================================

class DatasetProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)


# =============================================================================
# DATASET VISUALIZER (EDA)
# =============================================================================

class DatasetVisualizer:
    def __init__(self, X, y):
        self.df = pd.concat([X, y], axis=1)
        plt.style.use(STYLE)

    def eda_plots(self):
        sns.pairplot(self.df, hue="species", corner=True)
        plt.savefig(GRAPH_DIR + "pairplot.png", dpi=DPI)
        plt.close()

        self.df.hist(figsize=(12, 8))
        plt.savefig(GRAPH_DIR + "feature_histograms.png", dpi=DPI)
        plt.close()


# =============================================================================
# KNN TUNER
# =============================================================================

class KNNTuner:
    def tune_k(self, X_train, y_train, X_val, y_val):
        accuracies = []

        for k in K_RANGE:
            model = KNeighborsClassifier(
                n_neighbors=k,
                weights=WEIGHTS,
                metric=DISTANCE_METRIC
            )
            model.fit(X_train, y_train)
            acc = accuracy_score(y_val, model.predict(X_val))
            accuracies.append(acc)

        best_k = list(K_RANGE)[np.argmax(accuracies)]

        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(list(K_RANGE), accuracies, marker="o")
        plt.xlabel("K Value")
        plt.ylabel("Validation Accuracy")
        plt.title("Accuracy vs K")
        plt.savefig(GRAPH_DIR + "accuracy_vs_k.png", dpi=DPI)
        plt.close()

        return best_k


# =============================================================================
# KNN MODEL
# =============================================================================

class KNNModel:
    def __init__(self, k):
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            weights=WEIGHTS,
            metric=DISTANCE_METRIC
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:
    def evaluate(self, model, X, y, name):
        preds = model.predict(X)

        cm = confusion_matrix(y, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(GRAPH_DIR + f"confusion_matrix_{name.lower()}.png", dpi=DPI)
        plt.close()

        return ClassificationMetrics(
            accuracy=accuracy_score(y, preds),
            precision=precision_score(y, preds, average="macro"),
            recall=recall_score(y, preds, average="macro"),
            f1_score=f1_score(y, preds, average="macro")
        )


# =============================================================================
# DECISION BOUNDARY PLOT (PCA 2D)
# =============================================================================

def plot_decision_boundary(model, X, y):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    model_2d = KNeighborsClassifier(n_neighbors=model.model.n_neighbors)
    model_2d.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.title("KNN Decision Boundary (PCA Reduced)")
    plt.savefig(GRAPH_DIR + "decision_boundary.png", dpi=DPI)
    plt.close()


# =============================================================================
# ML PIPELINE
# =============================================================================

class MLPipeline:
    def run(self):
        X, y = DatasetLoader().load_data()

        DatasetVisualizer(X, y).eda_plots()

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        val_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            stratify=y_temp, random_state=RANDOM_STATE
        )

        processor = DatasetProcessor()
        X_train = processor.fit_transform(X_train)
        X_val = processor.transform(X_val)
        X_test = processor.transform(X_test)

        best_k = KNNTuner().tune_k(X_train, y_train, X_val, y_val)

        model = KNNModel(best_k)
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator()
        evaluator.evaluate(model, X_test, y_test, "test")

        plot_decision_boundary(model, X_train, y_train)

        joblib.dump(
            {"model": model, "scaler": processor},
            MODEL_SAVE_PATH
        )


def main():
    MLPipeline().run()


if __name__ == "__main__":
    main()
