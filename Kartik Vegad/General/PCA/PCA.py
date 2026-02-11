from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
VARIANCE_THRESHOLD = 0.95

ARTEFACT_PATH = "Unsupervised Learning/PCA/pca_transformer.pkl"

STYLE = "seaborn-v0_8-darkgrid"
DPI = 100


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class PCASummary:
    original_dim: int
    reduced_dim: int
    variance_retained: float
    reconstruction_error: float


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        iris = load_iris()
        x = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="species")
        return x, y


# ============================================================================
# SPLITTING
# ============================================================================

class DatasetSplitter:
    def split(self, x, y):
        x_temp, x_test, y_temp, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)

        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )

        return x_train, x_val, x_test, y_train, y_val, y_test


# ============================================================================
# TRANSFORMER
# ============================================================================

class PCATransformer:

    def __init__(self, variance_threshold=VARIANCE_THRESHOLD):
        self.scaler = StandardScaler()
        self.variance_threshold = variance_threshold
        self.pca: Optional[PCA] = None

    def fit(self, x_train: pd.DataFrame) -> None:
        scaled = self.scaler.fit_transform(x_train)

        temp = PCA().fit(scaled)
        cumulative = np.cumsum(temp.explained_variance_ratio_)
        n = np.argmax(cumulative >= self.variance_threshold) + 1

        self.pca = PCA(n_components=n)
        self.pca.fit(scaled)

        print(f"\n✓ PCA dimensions reduced: {scaled.shape[1]} → {n}")
        print(f"✓ Variance retained: {cumulative[n-1]:.4f}")

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.transform(x)
        return self.pca.transform(scaled)

    def inverse_transform(self, x_reduced: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(self.pca.inverse_transform(x_reduced))

    def explained_variance(self):
        return self.pca.explained_variance_ratio_

    def components(self):
        return self.pca.components_


# ============================================================================
# ANALYTICS
# ============================================================================

class PCAAnalytics:

    def summarize(
        self,
        transformer: PCATransformer,
        x_original: pd.DataFrame,
        x_reduced: np.ndarray
    ) -> PCASummary:

        reconstructed = transformer.inverse_transform(x_reduced)
        error = np.mean((x_original.values - reconstructed) ** 2)

        ratios = transformer.explained_variance()

        summary = PCASummary(
            original_dim=x_original.shape[1],
            reduced_dim=len(ratios),
            variance_retained=np.sum(ratios),
            reconstruction_error=error
        )

        print("\nPCA SUMMARY")
        print("Original dimensions:", summary.original_dim)
        print("Reduced dimensions:", summary.reduced_dim)
        print("Variance retained:", round(summary.variance_retained, 4))
        print("Reconstruction error:", round(summary.reconstruction_error, 6))

        return summary


# ============================================================================
# VISUALIZATION
# ============================================================================

class PCAVisualizer:

    def __init__(self, style="seaborn-v0_8-darkgrid", dpi=100):
        self.dpi = dpi
        plt.style.use(style)

    # --------------------------------------------------------------------- #
    # MASTER FUNCTION
    # --------------------------------------------------------------------- #
    def generate_all_reports(
        self,
        transformer,
        x_scaled: np.ndarray,
        y: Optional[np.ndarray],
        feature_names
    ):
        ratios = transformer.explained_variance()
        components = transformer.components()
        reduced = transformer.pca.transform(x_scaled)

        self._scree_plot(ratios)
        self._cumulative_plot(ratios)
        self._variance_bar(ratios)
        self._projection_2d(reduced, y)
        self._projection_3d(reduced, y)
        self._loadings_heatmap(components, feature_names)
        self._biplot(reduced, components, feature_names)
        self._reconstruction_error(transformer, x_scaled)

        print("✓ All PCA visual reports generated")

    # --------------------------------------------------------------------- #
    def _scree_plot(self, ratios):
        plt.figure(figsize=(8, 5), dpi=self.dpi)
        plt.plot(range(1, len(ratios) + 1), ratios, marker="o")
        plt.title("Scree Plot")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance")
        plt.tight_layout()
        plt.savefig("Unsupervised Learning/PCA/graphs/scree_plot.png")
        plt.close()

    # --------------------------------------------------------------------- #
    def _cumulative_plot(self, ratios):
        cumulative = np.cumsum(ratios)
        plt.figure(figsize=(8, 5), dpi=self.dpi)
        plt.plot(range(1, len(ratios) + 1), cumulative, marker="o")
        plt.axhline(0.95, linestyle="--")
        plt.title("Cumulative Variance")
        plt.xlabel("Components")
        plt.ylabel("Total Variance")
        plt.tight_layout()
        plt.savefig("Unsupervised Learning/PCA/graphs/cumulative_variance.png")
        plt.close()

    # --------------------------------------------------------------------- #
    def _variance_bar(self, ratios):
        plt.figure(figsize=(8, 5), dpi=self.dpi)
        plt.bar(range(1, len(ratios) + 1), ratios)
        plt.title("Variance Contribution")
        plt.tight_layout()
        plt.savefig("Unsupervised Learning/PCA/graphs/variance_bar.png")
        plt.close()

    # --------------------------------------------------------------------- #
    def _projection_2d(self, reduced, y):
        if reduced.shape[1] < 2:
            return

        plt.figure(figsize=(8, 6), dpi=self.dpi)

        if y is None:
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        else:
            for label in np.unique(y):
                mask = y == label
                plt.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    label=f"Class {label}",
                    alpha=0.7
                )
            plt.legend()

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("2D Projection")
        plt.tight_layout()
        plt.savefig("Unsupervised Learning/PCA/graphs/projection_2d.png")
        plt.close()

    # --------------------------------------------------------------------- #
    def _projection_3d(self, reduced, y):
        if reduced.shape[1] < 3:
            return

        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure(figsize=(8, 6), dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        if y is None:
            ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], alpha=0.7)
        else:
            for label in np.unique(y):
                mask = y == label
                ax.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    reduced[mask, 2],
                    label=f"Class {label}",
                    alpha=0.7
                )
            ax.legend()

        ax.set_title("3D Projection")
        plt.tight_layout()
        plt.savefig("Unsupervised Learning/PCA/graphs/projection_3d.png")
        plt.close()

    # --------------------------------------------------------------------- #
    def _loadings_heatmap(self, components, feature_names):
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        sns.heatmap(
            components,
            annot=True,
            cmap="coolwarm",
            xticklabels=feature_names,
            yticklabels=[f"PC{i+1}" for i in range(components.shape[0])]
        )
        plt.title("Component Loadings")
        plt.tight_layout()
        plt.savefig("Unsupervised Learning/PCA/graphs/loadings_heatmap.png")
        plt.close()

    # --------------------------------------------------------------------- #
    def _biplot(self, reduced, components, feature_names):
        if reduced.shape[1] < 2:
            return

        plt.figure(figsize=(8, 6), dpi=self.dpi)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.4)

        for i, feature in enumerate(feature_names):
            plt.arrow(
                0, 0,
                components[0, i] * 3,
                components[1, i] * 3
            )
            plt.text(
                components[0, i] * 3.2,
                components[1, i] * 3.2,
                feature
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Biplot")
        plt.tight_layout()
        plt.savefig("Unsupervised Learning/PCA/graphs/biplot.png")
        plt.close()

    # --------------------------------------------------------------------- #
    def _reconstruction_error(self, transformer, x_scaled):
        errors = []
        max_comp = getattr(transformer.pca, "n_features_in_", x_scaled.shape[1])

        for i in range(1, max_comp + 1):
            pca_temp = PCA(n_components=i)
            reduced = pca_temp.fit_transform(x_scaled)
            recon = pca_temp.inverse_transform(reduced)
            error = np.mean((x_scaled - recon) ** 2)
            errors.append(error)

        plt.figure(figsize=(8, 5), dpi=self.dpi)
        plt.plot(range(1, max_comp + 1), errors, marker="o")
        plt.title("Reconstruction Error")
        plt.xlabel("Components")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig("Unsupervised Learning/PCA/graphs/reconstruction_error.png")
        plt.close()

# ============================================================================
# PIPELINE
# ============================================================================

class Pipeline:
    def run(self):

        print("=" * 70)
        print("PCA FEATURE REDUCTION PIPELINE")
        print("=" * 70)

        # Load
        x, y = DataLoader().load()

        # Split
        splitter = DatasetSplitter()
        x_train, x_val, x_test, y_train, y_val, y_test = splitter.split(x, y)

        # Fit PCA only on training data
        transformer = PCATransformer()
        transformer.fit(x_train)

        # Transform all
        train_reduced = transformer.transform(x_train)
        val_reduced = transformer.transform(x_val)
        test_reduced = transformer.transform(x_test)

        # Analytics
        analytics = PCAAnalytics()
        analytics.summarize(transformer, x, transformer.transform(x))

        # Visuals
        visualizer = PCAVisualizer()
        x_scaled_full = transformer.scaler.transform(x)
        visualizer.generate_all_reports(
            transformer=transformer,
            x_scaled=x_scaled_full,
            y=y.values,
            feature_names=x.columns
        )

        # Save artefact
        joblib.dump(transformer, ARTEFACT_PATH)
        print(f"\n✓ PCA transformer saved → {ARTEFACT_PATH}")

        # Demonstrate reload
        loaded = joblib.load(ARTEFACT_PATH)
        sample = loaded.transform(x.iloc[:5])
        print("✓ Reloaded transformer works")


if __name__ == "__main__":
    Pipeline().run()
