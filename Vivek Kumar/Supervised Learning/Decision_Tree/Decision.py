import warnings
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ─── Global Variables ─────────────────────────────────────────────────────────
RANDOM_STATE   = 42
DATASET_PATH   = "drug200.csv"        # Multi-class classification dataset
TEST_SIZE      = 0.2
VALIDATION_SIZE = 0.1

# Decision Tree Hyperparameters
MAX_DEPTH       = None    # None → fully grown; set an int to limit depth
CRITERION       = 'gini'  # 'gini' | 'entropy' | 'log_loss'
SPLITTER        = 'best'  # 'best' | 'random'
MIN_SAMPLES_SPLIT = 2     # Minimum samples to split an internal node
MIN_SAMPLES_LEAF  = 1     # Minimum samples in a leaf node
MAX_FEATURES    = None    # None | 'sqrt' | 'log2' | int

# Model Persistence
MODEL_SAVE_PATH = "dt_model.pkl"

# Visualization
FIGURE_SIZE = (12, 8)
DPI         = 100
STYLE       = 'seaborn-v0_8-darkgrid'

# Target & categorical columns
TARGET_COLUMN       = 'Drug'
CATEGORICAL_COLUMNS = ['Sex', 'BP', 'Cholesterol']


# ─── Data Class ───────────────────────────────────────────────────────────────
@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics."""
    accuracy:  float
    precision: float
    recall:    float
    f1:        float

    def __str__(self) -> str:
        return (
            f"Model Performance Metrics:\n"
            f"{'=' * 50}\n"
            f"Accuracy:   {self.accuracy:.4f}\n"
            f"Precision:  {self.precision:.4f}\n"
            f"Recall:     {self.recall:.4f}\n"
            f"F1-Score:   {self.f1:.4f}\n"
            f"{'=' * 50}"
        )


# ─── DatasetLoader ────────────────────────────────────────────────────────────
class DatasetLoader:
    """Loads the Drug dataset from CSV or generates a synthetic fallback."""

    def __init__(self, dataset_path: str = None):
        self.dataset_path  = dataset_path
        self.data: Optional[pd.DataFrame]  = None
        self.target: Optional[pd.Series]   = None
        self.feature_names: Optional[list] = None

    def _generate_synthetic_dataset(self, n_samples: int = 200) -> Tuple[pd.DataFrame, pd.Series]:
        """Synthetic drug dataset with the same schema as the real CSV."""
        np.random.seed(RANDOM_STATE)

        ages         = np.random.randint(15, 75, n_samples)
        sexes        = np.random.choice(['M', 'F'], n_samples)
        bps          = np.random.choice(['LOW', 'NORMAL', 'HIGH'], n_samples)
        cholesterols = np.random.choice(['NORMAL', 'HIGH'], n_samples)
        na_to_k      = np.round(np.random.uniform(6.0, 38.0, n_samples), 3)

        # Rule-based drug assignment (mirrors real dataset logic)
        drugs = []
        for i in range(n_samples):
            if na_to_k > 20.0 if False else na_to_k[i] > 14.0:
                drugs.append('drugY')
            elif bps[i] == 'HIGH':
                drugs.append(np.random.choice(['drugA', 'drugB']))
            elif bps[i] == 'LOW':
                drugs.append(np.random.choice(['drugC', 'drugX']))
            else:
                drugs.append('drugX')

        df = pd.DataFrame({
            'Age': ages, 'Sex': sexes, 'BP': bps,
            'Cholesterol': cholesterols, 'Na_to_K': na_to_k
        })
        target = pd.Series(drugs, name=TARGET_COLUMN)
        self.feature_names = list(df.columns)
        self.data   = df
        self.target = target
        return df, target

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                if TARGET_COLUMN in df.columns:
                    self.data   = df.drop(columns=[TARGET_COLUMN])
                    self.target = df[TARGET_COLUMN]
                else:
                    self.data   = df.iloc[:, :-1]
                    self.target = df.iloc[:, -1]

                self.feature_names = list(self.data.columns)
                print(f"✓ Loaded dataset from: {self.dataset_path}")
                print(f"✓ Samples:  {len(self.data)}")
                print(f"✓ Features: {len(self.feature_names)}")
                return self.data, self.target
            except Exception as ex:
                print(f"⚠ Failed to load {self.dataset_path}: {ex}")
                print("⚠ Falling back to synthetic dataset generation")

        self.data, self.target = self._generate_synthetic_dataset()
        print(f"✓ Synthetic Drug dataset generated")
        print(f"✓ Samples:  {len(self.data)}")
        print(f"✓ Features: {', '.join(self.feature_names)}")
        print(f"✓ Target:   {TARGET_COLUMN} (multi-class)")
        return self.data, self.target


# ─── DatasetValidator ─────────────────────────────────────────────────────────
class DatasetValidator:
    """Validates and prints a summary of the loaded dataset."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        valid = True

        if self.data.empty or self.target.empty:
            print("✗ ERROR: Dataset is empty!")
            return False
        print("✓ Dataset is not empty")

        print(f"\n--- Shape ---")
        print(f"Features: {self.data.shape}  |  Target: {self.target.shape}")

        if self.data.shape[0] != self.target.shape[0]:
            print("✗ ERROR: Row count mismatch!")
            return False
        print("✓ Row counts match")

        print(f"\n--- Missing Values ---")
        miss_f = self.data.isnull().sum().sum()
        miss_t = self.target.isnull().sum()
        print(f"Features: {miss_f}  |  Target: {miss_t}")
        if miss_f > 0 or miss_t > 0:
            print("⚠ WARNING: Missing values detected")
            valid = False
        else:
            print("✓ No missing values")

        print(f"\n--- Data Types ---")
        print(self.data.dtypes)

        print(f"\n--- First 5 Rows ---")
        print(self.data.head())

        print(f"\n--- Statistical Summary ---")
        print(self.data.describe(include='all'))

        print(f"\n--- Class Distribution ---")
        cc = self.target.value_counts()
        print(cc)
        print(f"Number of classes: {cc.shape[0]}")

        return valid


# ─── DatasetProcessor ─────────────────────────────────────────────────────────
class DatasetProcessor:
    """
    Encodes categorical features and the target for Decision Tree training.

    ⓘ  Decision Trees can handle categorical data via Label Encoding.
       Unlike KNN, they are NOT sensitive to feature scale, so standardisation
       is NOT required.  We only encode categoricals → integers.
    """

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data.copy()
        self.target = target.copy()
        self.label_encoders: dict      = {}   # one LabelEncoder per categorical col
        self.target_encoder: LabelEncoder = LabelEncoder()
        self.class_names: list         = []
        self.processed_data: Optional[pd.DataFrame] = None
        self.processed_target: Optional[pd.Series]  = None

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        # --- Missing values ---
        print("\n--- Handling Missing Values ---")
        before = self.data.isnull().sum().sum()
        for col in self.data.columns:
            if self.data[col].dtype == object:
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            else:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
        self.target.fillna(self.target.mode()[0], inplace=True)
        after = self.data.isnull().sum().sum()
        print(f"Missing before: {before}  →  after: {after}")

        # --- Encode categoricals ---
        print("\n--- Label Encoding Categorical Features ---")
        print("  ⓘ  Decision Trees do not need standardisation;")
        print("     only categorical → integer encoding is required.")
        self.processed_data = self.data.copy()
        for col in CATEGORICAL_COLUMNS:
            if col in self.processed_data.columns:
                le = LabelEncoder()
                self.processed_data[col] = le.fit_transform(self.processed_data[col])
                self.label_encoders[col] = le
                print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # --- Encode target ---
        print("\n--- Encoding Target Variable ---")
        self.processed_target = pd.Series(
            self.target_encoder.fit_transform(self.target),
            name=TARGET_COLUMN
        )
        self.class_names = list(self.target_encoder.classes_)
        print(f"Classes: {dict(zip(self.class_names, self.target_encoder.transform(self.class_names)))}")

        print(f"\n--- Processed Shape ---")
        print(f"Features: {self.processed_data.shape}  |  Target: {self.processed_target.shape}")

        return self.processed_data, self.processed_target


# ─── DrugVisualizer ───────────────────────────────────────────────────────────
class DrugVisualizer:
    """Produces exploratory visualisations for the Drug dataset."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target
        plt.style.use(STYLE)

    def visualize(self):
        """Generate all exploratory plots used for dataset validation.

        Several of these graphs make it easy to spot problems:
        * imbalance in the target
        * missing values or unexpected nulls
        * distribution/outliers in numeric features
        * relationships between features (pairplot)
        """
        print(f"\n{'=' * 70}")
        print("DRUG DATASET VISUALIZATION")
        print(f"{'=' * 70}")

        # visual checks that are useful for validation
        self.plot_target_distribution()
        self.plot_missing_values_heatmap()
        self.plot_numerical_distributions()
        self.plot_pairwise_relationships()
        self.plot_categorical_vs_drug()
        self.plot_na_to_k_by_drug()
        self.plot_age_distribution()
        self.plot_correlation_heatmap()

        print("✓ All visualisations saved")

    # 1️⃣ Class distribution
    def plot_target_distribution(self):
        counts = self.target.value_counts().sort_values(ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = axes[0].bar(counts.index, counts.values, color=colors[:len(counts)],
                           edgecolor='black', alpha=0.8)
        for bar in bars:
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         str(int(bar.get_height())), ha='center', va='bottom')
        axes[0].set_title("Drug Class Distribution", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                    colors=colors[:len(counts)], startangle=90,
                    explode=[0.05]*len(counts))
        axes[1].set_title("Drug Class Ratio", fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig("drug_target_distribution.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Target distribution saved")

    # 2️⃣ Numerical feature distributions
    def plot_numerical_distributions(self):
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return
        fig, axes = plt.subplots(1, len(num_cols), figsize=(6 * len(num_cols), 5), dpi=DPI)
        if len(num_cols) == 1:
            axes = [axes]

        drug_classes = self.target.unique()
        cmap = plt.cm.get_cmap('tab10', len(drug_classes))

        for ax, col in zip(axes, num_cols):
            for idx, drug in enumerate(sorted(drug_classes)):
                subset = self.data[self.target == drug][col]
                ax.hist(subset, bins=20, alpha=0.6, label=drug,
                        color=cmap(idx), edgecolor='black')
            ax.set_title(f"{col} Distribution by Drug", fontsize=11, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("drug_numerical_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Numerical distributions saved")

    # 3️⃣ Categorical features vs Drug
    def plot_categorical_vs_drug(self):
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in self.data.columns]
        if not cat_cols:
            return

        fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 6), dpi=DPI)
        if len(cat_cols) == 1:
            axes = [axes]

        df_plot = self.data.copy()
        df_plot[TARGET_COLUMN] = self.target

        for ax, col in zip(axes, cat_cols):
            ct = pd.crosstab(df_plot[col], df_plot[TARGET_COLUMN])
            ct.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.8)
            ax.set_title(f"{col} vs Drug", fontsize=11, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=0)
            ax.legend(title='Drug', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("drug_categorical_vs_drug.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Categorical vs Drug plots saved")

    # 4️⃣ Na_to_K boxplot by drug
    def plot_na_to_k_by_drug(self):
        if 'Na_to_K' not in self.data.columns:
            return
        df_plot = self.data[['Na_to_K']].copy()
        df_plot[TARGET_COLUMN] = self.target

        plt.figure(figsize=(10, 6), dpi=DPI)
        sns.boxplot(data=df_plot, x=TARGET_COLUMN, y='Na_to_K', palette='tab10')
        plt.title("Na_to_K Ratio by Drug Type", fontsize=12, fontweight='bold')
        plt.xlabel("Drug")
        plt.ylabel("Na_to_K")
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("drug_na_to_k_boxplot.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Na_to_K boxplot saved")

    # 5️⃣ Age distribution by drug
    def plot_age_distribution(self):
        if 'Age' not in self.data.columns:
            return
        df_plot = self.data[['Age']].copy()
        df_plot[TARGET_COLUMN] = self.target
        drug_classes = sorted(self.target.unique())
        cmap = plt.cm.get_cmap('tab10', len(drug_classes))

        plt.figure(figsize=(12, 5), dpi=DPI)
        for idx, drug in enumerate(drug_classes):
            subset = df_plot[df_plot[TARGET_COLUMN] == drug]['Age']
            plt.hist(subset, bins=15, alpha=0.6, label=drug,
                     color=cmap(idx), edgecolor='black')
        plt.title("Age Distribution by Drug", fontsize=12, fontweight='bold')
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("drug_age_distribution.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Age distribution saved")

    # 6️⃣ Correlation heatmap (numeric only)
    def plot_correlation_heatmap(self):
        num_data = self.data.select_dtypes(include=[np.number])
        if num_data.shape[1] < 2:
            return
        plt.figure(figsize=(8, 6), dpi=DPI)
        sns.heatmap(num_data.corr(), annot=True, cmap='coolwarm', fmt='.2f',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Numerical Feature Correlation Heatmap", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("drug_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Correlation heatmap saved")

    # 0️⃣ Missing value heatmap (helps validation)
    def plot_missing_values_heatmap(self):
        if self.data.isnull().sum().sum() == 0:
            # still save an empty figure for consistency
            fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
            ax.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=12)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig("drug_missing_values.png", dpi=DPI, bbox_inches='tight')
            plt.close()
            print("[OK] Missing values plot saved (none present)")
            return

        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("drug_missing_values.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Missing values heatmap saved")

    # 0️⃣ Pairwise plots for numeric features
    def plot_pairwise_relationships(self):
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return
        sns.pairplot(pd.concat([self.data[num_cols], self.target.rename('Drug')], axis=1),
                     hue='Drug', corner=True, palette='tab10')
        plt.savefig("drug_pairplot.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Pairwise relationships plot saved")


# ─── DecisionTreeModel ────────────────────────────────────────────────────────
class DecisionTreeModel:
    """
    Decision Tree classifier wrapper for multi-class drug prediction.

    Key difference from KNN
    ───────────────────────
    • KNN is a lazy learner — it memorises the training set and calculates
      distances at prediction time. It is highly sensitive to feature scale.
    • Decision Tree is an eager learner — it builds a tree of if/else rules
      during training.  It is completely scale-invariant and naturally handles
      categorical (encoded) features.
    """

    def __init__(
        self,
        max_depth: Optional[int]  = MAX_DEPTH,
        criterion: str            = CRITERION,
        splitter: str             = SPLITTER,
        min_samples_split: int    = MIN_SAMPLES_SPLIT,
        min_samples_leaf: int     = MIN_SAMPLES_LEAF,
        max_features              = MAX_FEATURES,
        class_names: list         = None,
        feature_names: list       = None,
    ):
        self.max_depth         = max_depth
        self.criterion         = criterion
        self.class_names       = class_names or []
        self.feature_names     = feature_names or []

        self.model = DecisionTreeClassifier(
            max_depth         = max_depth,
            criterion         = criterion,
            splitter          = splitter,
            min_samples_split = min_samples_split,
            min_samples_leaf  = min_samples_leaf,
            max_features      = max_features,
            random_state      = RANDOM_STATE,
        )

    # Training
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"\n{'=' * 60}")
        print("TRAINING DECISION TREE MODEL (DRUG CLASSIFICATION)")
        print(f"{'=' * 60}")
        print(f"Criterion : {self.criterion}")
        print(f"Max Depth : {self.max_depth if self.max_depth else 'None (fully grown)'}")

        self.model.fit(X_train, y_train)
        print(f"✓ Training completed")
        print(f"  Tree depth  : {self.model.get_depth()}")
        print(f"  Leaf nodes  : {self.model.get_n_leaves()}")

        try:
            joblib.dump(self.model, MODEL_SAVE_PATH)
            print(f"✓ Model saved → {MODEL_SAVE_PATH}")
        except Exception as ex:
            print(f"⚠ Could not save model: {ex}")

    # Predictions
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    # Evaluation
    def evaluate(self, X: np.ndarray, y: np.ndarray, name: str = "Test Set"):
        print(f"\n{'=' * 60}")
        print(f"DECISION TREE EVALUATION — {name}")
        print(f"{'=' * 60}")

        y_pred = self.predict(X)

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y, y_pred, average='weighted', zero_division=0)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f} (weighted)")
        print(f"Recall    : {rec:.4f} (weighted)")
        print(f"F1-Score  : {f1:.4f} (weighted)")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y, y_pred))

        print("\nClassification Report:")
        print(classification_report(y, y_pred,
                                    target_names=self.class_names,
                                    zero_division=0))
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Feature importance
    def get_feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)


# ─── ModelEvaluator ───────────────────────────────────────────────────────────
class ModelEvaluator:
    """Generates evaluation plots for the Decision Tree model."""

    def __init__(self, model: DecisionTreeModel):
        self.model = model

    def evaluate(self, X, y_true, dataset_name="Dataset"):
        print(f"\n{'=' * 70}")
        print(f"MODEL EVALUATION — {dataset_name}")
        print(f"{'=' * 70}")

        y_pred      = self.model.predict(X)
        y_pred_prob = self.model.predict_proba(X)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f} (weighted)")
        print(f"Recall    : {rec:.4f} (weighted)")
        print(f"F1-Score  : {f1:.4f} (weighted)")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=self.model.class_names,
                                    zero_division=0))

        self._plot_evaluation(y_true, y_pred, dataset_name, acc, prec, rec, f1)
        self._plot_feature_importance(dataset_name)

    def _plot_evaluation(self, y_true, y_pred, dataset_name, acc, prec, rec, f1):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=DPI)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=self.model.class_names,
                    yticklabels=self.model.class_names,
                    cbar_kws={"shrink": 0.8})
        axes[0].set_title(f'Confusion Matrix — {dataset_name}', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # Metrics bar chart
        metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [acc, prec, rec, f1]
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = axes[1].bar(metrics_names, metrics_values, color=colors_bar,
                           edgecolor='black', alpha=0.8)
        axes[1].set_ylim([0, 1.05])
        axes[1].set_ylabel('Score')
        axes[1].set_title(f'Performance Metrics — {dataset_name}', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, metrics_values):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        fname = f'evaluation_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Evaluation plot saved → {fname}")

    def _plot_feature_importance(self, dataset_name):
        importances = self.model.get_feature_importances()
        colors = ['#d62728' if x == importances.max() else '#1f77b4' for x in importances.values]

        plt.figure(figsize=(10, 5), dpi=DPI)
        bars = plt.barh(importances.index, importances.values, color=colors,
                        edgecolor='black', alpha=0.8)
        for bar, val in zip(bars, importances.values):
            plt.text(val + 0.005, bar.get_y() + bar.get_height()/2.,
                     f'{val:.4f}', va='center', fontsize=9)
        plt.xlabel('Feature Importance (Gini / Entropy Gain)', fontsize=11)
        plt.title(f'Decision Tree Feature Importance — {dataset_name}',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig("drug_feature_importance.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature importance plot saved")


# ─── MLPipeline ───────────────────────────────────────────────────────────────
class MLPipeline:
    """End-to-end Decision Tree pipeline for Drug Classification."""

    def __init__(self):
        self.loader    = DatasetLoader(DATASET_PATH)
        self.processor = None
        self.model     = None
        self.evaluator = None

    def run(self):
        print(f"\n{'=' * 70}")
        print("DECISION TREE PIPELINE — DRUG CLASSIFICATION")
        print(f"{'=' * 70}")

        # 1️⃣ Load
        data, target = self.loader.load_data()

        # 2️⃣ Visualise raw data (useful for validating dataset)
        visualizer = DrugVisualizer(data, target)
        visualizer.visualize()

        # 3️⃣ Validate
        validator = DatasetValidator(data, target)
        validator.verify_dataset()

        # 4️⃣ Process (encode categoricals)
        self.processor = DatasetProcessor(data, target)
        proc_data, proc_target = self.processor.process_dataset()

        # 5️⃣ Split
        X_train, X_test, y_train, y_test = train_test_split(
            proc_data.values,
            proc_target.values,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
            stratify     = proc_target,
        )
        print(f"\n✓ Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")

        # 6️⃣ Train
        self.model = DecisionTreeModel(
            class_names   = self.processor.class_names,
            feature_names = list(proc_data.columns),
        )
        self.model.fit(X_train, y_train)

        # 7️⃣ Evaluate
        self.evaluator = ModelEvaluator(self.model)
        self.evaluator.evaluate(X_train, y_train, "Training Set")
        self.evaluator.evaluate(X_test,  y_test,  "Test Set")

        # 8️⃣ Cross-Validation
        self._perform_cross_validation(proc_data.values, proc_target.values)

        # 9️⃣ Visualise the tree
        self._plot_decision_tree(list(proc_data.columns))

        # 🔟 Predict a new patient
        self._predict_new_patient(proc_data.columns)

    # ── Cross-validation ──────────────────────────────────────────────────────
    def _perform_cross_validation(self, X, y):
        print(f"\n{'=' * 70}")
        print("K-FOLD CROSS-VALIDATION (5-Fold)")
        print(f"{'=' * 70}")

        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        clf = DecisionTreeClassifier(criterion=CRITERION, random_state=RANDOM_STATE)

        acc_scores  = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        f1_scores   = cross_val_score(clf, X, y, cv=cv, scoring='f1_weighted')
        prec_scores = cross_val_score(clf, X, y, cv=cv, scoring='precision_weighted')
        rec_scores  = cross_val_score(clf, X, y, cv=cv, scoring='recall_weighted')

        print(f"\nAccuracy  : {acc_scores.mean():.4f}  (+/- {acc_scores.std():.4f})")
        print(f"Precision : {prec_scores.mean():.4f}  (+/- {prec_scores.std():.4f})")
        print(f"Recall    : {rec_scores.mean():.4f}  (+/- {rec_scores.std():.4f})")
        print(f"F1-Score  : {f1_scores.mean():.4f}  (+/- {f1_scores.std():.4f})")

        generalisation = "GOOD generalisation ✓" if acc_scores.std() < 0.05 else "HIGH variance ⚠"
        print(f"\n✓ Cross-validation indicates {generalisation}")

        # Plot CV accuracy per fold
        plt.figure(figsize=(8, 4), dpi=DPI)
        plt.plot(range(1, 6), acc_scores, 'o-', linewidth=2, markersize=8, label='Fold Accuracy')
        plt.axhline(acc_scores.mean(), color='red', linestyle='--',
                    label=f'Mean ({acc_scores.mean():.4f})')
        plt.fill_between(range(1, 6),
                         acc_scores.mean() - acc_scores.std(),
                         acc_scores.mean() + acc_scores.std(),
                         alpha=0.2, color='blue')
        plt.title("5-Fold Cross-Validation Accuracy", fontsize=12, fontweight='bold')
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.xticks(range(1, 6))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("drug_cross_validation.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Cross-validation plot saved")

    # ── Visualise the tree ────────────────────────────────────────────────────
    def _plot_decision_tree(self, feature_names: list):
        print(f"\n{'=' * 70}")
        print("DECISION TREE VISUALISATION")
        print(f"{'=' * 70}")

        # ── Text rules ────────────────────────────────────────────────────────
        rules = export_text(
            self.model.model,
            feature_names=feature_names,
            max_depth=4
        )
        print("\n--- Decision Rules (first 4 levels) ---")
        print(rules)

        # ── Graphical tree (limited depth for readability) ────────────────────
        plt.figure(figsize=(22, 10), dpi=DPI)
        plot_tree(
            self.model.model,
            feature_names = feature_names,
            class_names   = self.processor.class_names,
            filled        = True,
            rounded       = True,
            fontsize      = 9,
            max_depth     = 4,
            impurity      = True,
            proportion    = False,
        )
        plt.title("Decision Tree — Drug Classification (max display depth = 4)",
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("drug_decision_tree.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Decision tree diagram saved → drug_decision_tree.png")

        # ── Feature importance bar ─────────────────────────────────────────────
        importances = self.model.get_feature_importances()
        plt.figure(figsize=(10, 5), dpi=DPI)
        colors = ['#d62728' if x == importances.max() else '#1f77b4' for x in importances.values]
        bars = plt.barh(importances.index, importances.values, color=colors,
                        edgecolor='black', alpha=0.8)
        for bar, val in zip(bars, importances.values):
            plt.text(val + 0.005, bar.get_y() + bar.get_height()/2.,
                     f'{val:.4f}', va='center', fontsize=9)
        plt.xlabel("Gini Importance")
        plt.title("Feature Importances — Drug Decision Tree", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig("drug_feature_importance_bar.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature importance bar saved")

    # ── New patient prediction ────────────────────────────────────────────────
    def _predict_new_patient(self, feature_cols):
        print(f"\n{'=' * 70}")
        print("NEW PATIENT PREDICTION")
        print(f"{'=' * 70}")

        # Example patient — matches the dataset schema
        raw_patient = {
            'Age': 45,
            'Sex': 'M',
            'BP': 'HIGH',
            'Cholesterol': 'NORMAL',
            'Na_to_K': 16.5,
        }
        print("Patient profile:")
        for k, v in raw_patient.items():
            print(f"  {k}: {v}")

        # Encode categoricals using the fitted label encoders
        encoded = {}
        for col in feature_cols:
            val = raw_patient[col]
            if col in self.processor.label_encoders:
                le = self.processor.label_encoders[col]
                try:
                    encoded[col] = le.transform([val])[0]
                except ValueError:
                    print(f"  ⚠ Unknown category '{val}' for {col}; defaulting to 0")
                    encoded[col] = 0
            else:
                encoded[col] = val

        X_new = np.array([[encoded[col] for col in feature_cols]])
        pred_encoded = self.model.predict(X_new)[0]
        pred_drug    = self.processor.target_encoder.inverse_transform([pred_encoded])[0]
        probabilities = self.model.predict_proba(X_new)[0]

        print(f"\nPredicted Drug  : {pred_drug}")
        print(f"\nClass Probabilities:")
        for cls, prob in zip(self.processor.class_names, probabilities):
            bar = '█' * int(prob * 30)
            print(f"  {cls:<10} {prob:.4f}  {bar}")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    """Run the Decision Tree Drug Classification pipeline."""
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n❌ ERROR during pipeline execution:")
        print(f"{type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    main()