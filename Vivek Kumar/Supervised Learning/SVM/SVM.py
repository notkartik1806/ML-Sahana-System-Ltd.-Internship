import warnings
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import seaborn as sns
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ─── Global Variables ────────────────────────────────────────────────────────
RANDOM_STATE   = 42
DATASET_PATH   = "loan_data.csv"      # Binary classification dataset
TEST_SIZE      = 0.2
VALIDATION_SIZE = 0.1

# SVM Hyperparameters
SVM_KERNEL     = 'rbf'        # 'linear' | 'rbf' | 'poly' | 'sigmoid'
SVM_C          = 1.0          # Regularization strength (higher → less regularization)
SVM_GAMMA      = 'scale'      # Kernel coefficient: 'scale' | 'auto' | float
SVM_DEGREE     = 3            # Degree for 'poly' kernel only
SVM_PROBABILITY = True        # Required for ROC-AUC (enables predict_proba)

# SVM Tuning (GridSearchCV)
PARAM_GRID = {
    'C':      [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma':  ['scale', 'auto'],
}
TUNE_CV   = 3      # Cross-validation folds during GridSearchCV
TUNE_JOBS = -1     # Parallel jobs (-1 = all CPUs)

# Model Persistence
MODEL_SAVE_PATH = "svm_model.pkl"

# Visualization
FIGURE_SIZE = (12, 8)
DPI         = 100
STYLE       = 'seaborn-v0_8-darkgrid'

# Feature columns and target
CATEGORICAL_COLS = [
    'person_gender', 'person_education',
    'person_home_ownership', 'loan_intent',
    'previous_loan_defaults_on_file'
]
TARGET_COL = 'loan_status'


# ─── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics."""
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    roc_auc:   float

    def __str__(self) -> str:
        return (
            f"Model Performance Metrics:\n"
            f"{'=' * 50}\n"
            f"Accuracy:   {self.accuracy:.4f}\n"
            f"Precision:  {self.precision:.4f}\n"
            f"Recall:     {self.recall:.4f}\n"
            f"F1-Score:   {self.f1:.4f}\n"
            f"ROC-AUC:    {self.roc_auc:.4f}\n"
            f"{'=' * 50}"
        )


# ─── Dataset Loader ───────────────────────────────────────────────────────────
class DatasetLoader:
    """Loads the loan-approval CSV or falls back to a synthetic dataset."""

    def __init__(self, dataset_path: str = None):
        self.dataset_path  = dataset_path
        self.data: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series]  = None
        self.feature_names: Optional[list] = None

    # ── Synthetic fallback ─────────────────────────────────────────────────
    def _generate_synthetic_dataset(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        np.random.seed(RANDOM_STATE)
        n_approved   = n_samples // 2
        n_rejected   = n_samples - n_approved

        genders    = ['male', 'female']
        educations = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
        ownerships = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
        intents    = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
        defaults   = ['Yes', 'No']

        def make_records(n, approved: bool) -> dict:
            age     = np.random.randint(20, 70, n).astype(float)
            income  = np.random.normal(70000 if approved else 35000, 20000, n).clip(10000, 200000)
            emp_exp = np.random.randint(0, 15, n).astype(float)
            credit  = np.random.normal(680 if approved else 540, 60, n).clip(300, 850).astype(int)
            loan_a  = np.random.normal(8000 if approved else 20000, 5000, n).clip(500, 35000)
            rate    = np.random.normal(10.5 if approved else 15.5, 3, n).clip(5, 24)
            pct_inc = (loan_a / income).clip(0.01, 0.99)
            hist    = np.random.randint(1, 15, n).astype(float)
            return {
                'person_age':              age,
                'person_gender':           np.random.choice(genders, n),
                'person_education':        np.random.choice(educations, n),
                'person_income':           income,
                'person_emp_exp':          emp_exp,
                'person_home_ownership':   np.random.choice(ownerships, n),
                'loan_amnt':               loan_a,
                'loan_intent':             np.random.choice(intents, n),
                'loan_int_rate':           rate,
                'loan_percent_income':     pct_inc,
                'cb_person_cred_hist_length': hist,
                'credit_score':            credit,
                'previous_loan_defaults_on_file':
                    np.random.choice(['No', 'No', 'No', 'Yes'] if approved else ['Yes', 'Yes', 'No'], n),
            }

        rec0 = make_records(n_approved, True)
        rec1 = make_records(n_rejected, False)

        combined = {k: np.concatenate([rec0[k], rec1[k]]) for k in rec0}
        self.data   = pd.DataFrame(combined)
        self.target = pd.Series(
            np.concatenate([np.ones(n_approved), np.zeros(n_rejected)]),
            name=TARGET_COL
        ).astype(int)

        idx = np.random.permutation(n_samples)
        self.data   = self.data.iloc[idx].reset_index(drop=True)
        self.target = self.target.iloc[idx].reset_index(drop=True)
        self.feature_names = list(self.data.columns)
        return self.data, self.target

    # ── Main loader ────────────────────────────────────────────────────────
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                if TARGET_COL in df.columns:
                    self.data   = df.drop(columns=[TARGET_COL])
                    self.target = df[TARGET_COL]
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
                print("⚠ Falling back to synthetic dataset")

        self.data, self.target = self._generate_synthetic_dataset()
        print(f"✓ Synthetic loan dataset generated")
        print(f"✓ Samples:  {len(self.data)}")
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ Feature names: {', '.join(self.feature_names)}")
        print(f"✓ Classes: 0 (Rejected), 1 (Approved)")
        return self.data, self.target


# ─── Dataset Validator ────────────────────────────────────────────────────────
class DatasetValidator:
    """Validates the raw dataset before processing."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        ok = True
        if self.data.empty or self.target.empty:
            print("✗ ERROR: Dataset is empty!")
            return False
        print("✓ Dataset is not empty")

        print(f"\n--- Shape ---")
        print(f"Features : {self.data.shape}")
        print(f"Target   : {self.target.shape}")

        if self.data.shape[0] != self.target.shape[0]:
            print("✗ ERROR: Row mismatch between features and target!")
            return False
        print("✓ Row counts match")

        print(f"\n--- Missing Values ---")
        miss_f = self.data.isnull().sum().sum()
        miss_t = self.target.isnull().sum()
        print(f"Features: {miss_f}")
        print(f"Target  : {miss_t}")
        if miss_f > 0 or miss_t > 0:
            print("⚠ WARNING: Missing values present")
            ok = False
        else:
            print("✓ No missing values")

        print(f"\n--- Column Data Types ---")
        print(self.data.dtypes)

        print(f"\n--- First 5 Rows ---")
        print(self.data.head())

        print(f"\n--- Statistical Summary (Numeric) ---")
        print(self.data.describe())

        print(f"\n--- Class Distribution ---")
        counts = self.target.value_counts()
        print(counts)
        print(f"Balance ratio: {counts.min() / counts.max():.2f}")

        inf_cnt = np.isinf(self.data.select_dtypes(include=[np.number]).values).sum()
        if inf_cnt > 0:
            print(f"⚠ WARNING: {inf_cnt} infinite values")
            ok = False
        else:
            print("✓ No infinite values")

        return ok


# ─── Dataset Processor ────────────────────────────────────────────────────────
class DatasetProcessor:
    """
    Encodes categoricals, imputes missing values, and standardizes features.

    SVM-SPECIFIC NOTE:
    ──────────────────
    SVMs are EXTREMELY sensitive to feature scale. The kernel function (especially
    RBF) computes distances in feature space, so unscaled features with large
    ranges (e.g. person_income ~70 000) dominate over small-range features
    (e.g. loan_percent_income ~0.4). Z-score standardization is mandatory.
    """

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data             = data.copy()
        self.target           = target.copy()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_means    = None
        self.feature_stds     = None
        self.processed_data: Optional[pd.DataFrame]  = None
        self.processed_target: Optional[pd.Series]   = None
        self.feature_names: Optional[list]            = None

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        # ── 1. Encode categorical columns ──────────────────────────────────
        print("\n--- Encoding Categorical Features ---")
        cat_cols = [c for c in CATEGORICAL_COLS if c in self.data.columns]
        for col in cat_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
            print(f"  ✓ Encoded '{col}'  →  classes: {list(le.classes_)}")

        # ── 2. Impute missing values ───────────────────────────────────────
        print("\n--- Handling Missing Values ---")
        miss_before = self.data.isnull().sum().sum()
        for col in self.data.columns:
            if self.data[col].isnull().any():
                if self.data[col].dtype in [np.float64, np.int64]:
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                else:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        self.target.fillna(self.target.mode()[0], inplace=True)
        miss_after = self.data.isnull().sum().sum()
        print(f"Missing before: {miss_before}  →  after: {miss_after}")

        # ── 3. Standardize (Z-score) ───────────────────────────────────────
        print("\n--- Feature Standardization (Z-Score) ---")
        print("  ⓘ  SVM requires scaled features — the RBF kernel uses")
        print("     Euclidean distances; unscaled data causes the SVM")
        print("     to ignore small-range features entirely.")
        self.feature_means  = self.data.mean()
        self.feature_stds   = self.data.std().replace(0, 1)          # avoid ÷0
        self.processed_data = (self.data - self.feature_means) / self.feature_stds
        self.feature_names  = list(self.processed_data.columns)
        print("✓ Features standardized (mean=0, std=1)")

        # ── 4. Binarize target ─────────────────────────────────────────────
        print("\n--- Target Encoding ---")
        unique = self.target.unique()
        if len(unique) > 2:
            print("⚠ More than 2 classes — converting to binary via median split")
            self.processed_target = (self.target > self.target.median()).astype(int)
        else:
            self.processed_target = self.target.astype(int)
        print(f"Target classes: {sorted(self.processed_target.unique())} "
              f"(0=Rejected, 1=Approved)")

        print(f"\n--- Processed Shape ---")
        print(f"Features : {self.processed_data.shape}")
        print(f"Target   : {self.processed_target.shape}")

        return self.processed_data, self.processed_target


# ─── Loan Visualizer ──────────────────────────────────────────────────────────
class LoanVisualizer:
    """Generates EDA visualizations for the Loan Approval dataset."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target
        plt.style.use(STYLE)

    def visualize(self):
        print(f"\n{'=' * 70}")
        print("LOAN DATASET VISUALIZATION")
        print(f"{'=' * 70}")

        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_numeric_distributions()
        self.plot_categorical_vs_target()
        self.plot_feature_boxplots()
        self.plot_top_correlations()
        self.plot_feature_violin_plots()
        self.plot_feature_kde_plots()
        self.plot_feature_importance()
        self.plot_3d_scatter()

        print("✓ All loan visualizations saved")

    # 1️⃣  Class distribution
    def plot_target_distribution(self):
        counts = self.target.value_counts().sort_index()
        labels = ['Rejected (0)', 'Approved (1)']
        colors = ['#e74c3c', '#2ecc71']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='black', alpha=0.75)
        axes[0].set_title('Loan Status Distribution', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3, axis='y')
        for b in bars:
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                         str(int(b.get_height())), ha='center', va='bottom')

        axes[1].pie(counts.values, labels=['Rejected', 'Approved'],
                    autopct='%1.1f%%', colors=colors,
                    startangle=90, explode=(0.05, 0.05))
        axes[1].set_title('Class Ratio', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig("loan_target_distribution.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Target distribution saved")

    # 2️⃣  Correlation heatmap
    def plot_correlation_heatmap(self):
        df = self.data.select_dtypes(include=[np.number]).copy()
        df[TARGET_COL] = self.target

        plt.figure(figsize=(14, 10), dpi=DPI)
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm',
                    fmt='.2f', center=0, square=True,
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Loan Dataset Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("loan_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Correlation heatmap saved")

    # 3️⃣  Numeric distributions by class
    def plot_numeric_distributions(self):
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        cols, rows = 3, (len(num_cols) + 2) // 3

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            axes[i].hist(self.data[self.target == 0][col], bins=30,
                         alpha=0.6, label='Rejected', color='#e74c3c', edgecolor='black')
            axes[i].hist(self.data[self.target == 1][col], bins=30,
                         alpha=0.6, label='Approved', color='#2ecc71', edgecolor='black')
            axes[i].set_title(col, fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        for idx in range(len(num_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("loan_numeric_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Numeric distributions saved")

    # 4️⃣  Categorical vs target (bar charts)
    def plot_categorical_vs_target(self):
        cat_cols = [c for c in CATEGORICAL_COLS if c in self.data.columns]
        if not cat_cols:
            return

        cols, rows = 2, (len(cat_cols) + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            ct = pd.crosstab(self.data[col], self.target, normalize='index') * 100
            ct.columns = ['Rejected %', 'Approved %'] if len(ct.columns) == 2 else ct.columns
            ct.plot(kind='bar', ax=axes[i], color=['#e74c3c', '#2ecc71'],
                    edgecolor='black', alpha=0.75)
            axes[i].set_title(f'{col} vs Loan Status', fontsize=11, fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Percentage (%)')
            axes[i].legend(loc='upper right')
            axes[i].tick_params(axis='x', rotation=30)
            axes[i].grid(True, alpha=0.3, axis='y')

        for idx in range(len(cat_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("loan_categorical_vs_target.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Categorical vs target plot saved")

    # 5️⃣  Boxplots for top correlated numeric features
    def plot_feature_boxplots(self):
        num_data = self.data.select_dtypes(include=[np.number])
        corr     = num_data.corrwith(self.target).abs().sort_values(ascending=False)
        top_feats = corr.head(4).index

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        for i, feat in enumerate(top_feats):
            sns.boxplot(x=self.target, y=self.data[feat], ax=axes[i],
                        palette=['#e74c3c', '#2ecc71'])
            axes[i].set_title(f'{feat} vs Loan Status (Corr: {corr[feat]:.3f})',
                              fontsize=11, fontweight='bold')
            axes[i].set_xticklabels(['Rejected', 'Approved'])
            axes[i].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("loan_feature_boxplots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature boxplots saved")

    # 6️⃣  Scatter plots for top correlations
    def plot_top_correlations(self):
        num_data  = self.data.select_dtypes(include=[np.number])
        corr      = num_data.corrwith(self.target).abs().sort_values(ascending=False)
        top_feats = corr.head(4).index.tolist()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        for idx, feat in enumerate(top_feats):
            c0 = self.data[self.target == 0]
            c1 = self.data[self.target == 1]
            axes[idx].scatter(c0[feat], [0] * len(c0), alpha=0.5, s=40,
                              label='Rejected', color='#e74c3c')
            axes[idx].scatter(c1[feat], [1] * len(c1), alpha=0.5, s=40,
                              label='Approved', color='#2ecc71')
            axes[idx].set_xlabel(feat)
            axes[idx].set_ylabel('Class')
            axes[idx].set_yticks([0, 1])
            axes[idx].set_yticklabels(['Rejected', 'Approved'])
            axes[idx].set_title(f'{feat} vs Outcome (Corr: {corr[feat]:.3f})',
                                fontsize=11, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("loan_top_correlations.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Top correlations scatter saved")

    # 7️⃣  Violin plots
    def plot_feature_violin_plots(self):
        num_cols  = self.data.select_dtypes(include=[np.number]).columns.tolist()
        cols, rows = 3, (len(num_cols) + 2) // 3

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4), dpi=DPI)
        axes = axes.flatten()

        plot_data             = self.data.select_dtypes(include=[np.number]).copy()
        plot_data[TARGET_COL] = self.target

        for i, col in enumerate(num_cols):
            sns.violinplot(data=plot_data, x=TARGET_COL, y=col, ax=axes[i],
                           palette=['#e74c3c', '#2ecc71'], alpha=0.7)
            axes[i].set_title(f'{col} by Loan Status', fontsize=11, fontweight='bold')
            axes[i].set_xticklabels(['Rejected', 'Approved'])
            axes[i].grid(True, alpha=0.3, axis='y')

        for idx in range(len(num_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("loan_violin_plots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Violin plots saved")

    # 8️⃣  KDE plots for top numeric features
    def plot_feature_kde_plots(self):
        num_data  = self.data.select_dtypes(include=[np.number])
        corr      = num_data.corrwith(self.target).abs().sort_values(ascending=False)
        top_feats = corr.head(min(8, len(corr))).index.tolist()

        cols, rows = 2, (len(top_feats) + 1) // 2
        fig, axes  = plt.subplots(rows, cols, figsize=(15, rows * 4), dpi=DPI)
        axes       = axes.flatten()

        for i, feat in enumerate(top_feats):
            c0 = self.data[self.target == 0][feat].dropna()
            c1 = self.data[self.target == 1][feat].dropna()
            c0.plot.kde(ax=axes[i], linewidth=2, label='Rejected',  color='#e74c3c')
            c1.plot.kde(ax=axes[i], linewidth=2, label='Approved',  color='#2ecc71')
            axes[i].set_title(f'{feat} — KDE', fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        for idx in range(len(top_feats), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("loan_kde_plots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] KDE plots saved")

    # 9️⃣  Feature importance (correlation + variance)
    def plot_feature_importance(self):
        num_data = self.data.select_dtypes(include=[np.number])
        corr     = num_data.corrwith(self.target).abs().sort_values(ascending=False)
        var_imp  = (num_data[self.target == 0].var() + num_data[self.target == 1].var()
                    ).sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=DPI)

        c_corr = ['#d62728' if x == corr.max() else '#1f77b4' for x in corr.values]
        axes[0].barh(corr.index, corr.values, color=c_corr, edgecolor='black', alpha=0.75)
        axes[0].set_xlabel('|Correlation| with Target')
        axes[0].set_title('Feature Importance — Correlation', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(corr.values):
            axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

        c_var = ['#d62728' if x == var_imp.max() else '#2ca02c' for x in var_imp.values]
        axes[1].barh(var_imp.index, var_imp.values, color=c_var, edgecolor='black', alpha=0.75)
        axes[1].set_xlabel('Variance (sum of classes)')
        axes[1].set_title('Feature Importance — Variance', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig("loan_feature_importance.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature importance saved")

    # 🔟  3D scatter (top-3 numeric features)
    def plot_3d_scatter(self):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        num_data  = self.data.select_dtypes(include=[np.number])
        corr      = num_data.corrwith(self.target).abs().sort_values(ascending=False)
        top3      = corr.head(3).index.tolist()
        if len(top3) < 3:
            print("[OK] Skipping 3D scatter (fewer than 3 numeric features)")
            return

        fig = plt.figure(figsize=(12, 9), dpi=DPI)
        ax  = fig.add_subplot(111, projection='3d')

        for cls, label, color in [(0, 'Rejected', '#e74c3c'), (1, 'Approved', '#2ecc71')]:
            mask = self.target == cls
            ax.scatter(self.data[mask][top3[0]], self.data[mask][top3[1]],
                       self.data[mask][top3[2]],
                       c=color, label=label, s=25, alpha=0.6, edgecolors='none')

        ax.set_xlabel(top3[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(top3[1], fontsize=10, fontweight='bold')
        ax.set_zlabel(top3[2], fontsize=10, fontweight='bold')
        ax.set_title('3D Scatter — Top 3 Features', fontsize=13, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig("loan_3d_scatter.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 3D scatter saved")


# ─── SVM Model ────────────────────────────────────────────────────────────────
class SVMModel:
    """
    Support Vector Machine for Loan Approval Prediction.

    SVM Theory (brief):
    ───────────────────
    SVM finds the hyperplane that maximises the margin between the two
    classes.  Only the closest training points (support vectors) influence
    the boundary.

    Key hyperparameters
    ───────────────────
    C        – Regularisation: small C → wide margin, some misclassifications
               (better generalisation); large C → tight margin, few training
               errors (risk of overfitting).
    kernel   – Maps data to higher-dimensional space so a linear hyperplane
               can separate non-linear data.  'rbf' is the default choice.
    gamma    – RBF bandwidth: 'scale' = 1/(n_features * X.var()), a safe
               default that automatically adapts to the data.
    """

    def __init__(self,
                 kernel: str  = SVM_KERNEL,
                 C: float     = SVM_C,
                 gamma        = SVM_GAMMA,
                 degree: int  = SVM_DEGREE,
                 probability: bool = SVM_PROBABILITY):
        self.kernel      = kernel
        self.C           = C
        self.gamma       = gamma
        self.degree      = degree
        self.probability = probability
        self.model       = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            random_state=RANDOM_STATE,
        )
        self.is_fitted = False

    # ── Training ───────────────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"\n{'=' * 60}")
        print("TRAINING SVM MODEL (LOAN APPROVAL)")
        print(f"{'=' * 60}")
        print(f"  Kernel  : {self.kernel}")
        print(f"  C       : {self.C}")
        print(f"  Gamma   : {self.gamma}")
        if self.kernel == 'poly':
            print(f"  Degree  : {self.degree}")

        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("✓ SVM training completed")

        if hasattr(self.model, 'support_vectors_'):
            print(f"✓ Support vectors: {len(self.model.support_vectors_)}")

        try:
            joblib.dump(self.model, MODEL_SAVE_PATH)
            print(f"✓ Model saved → {MODEL_SAVE_PATH}")
        except Exception as ex:
            print(f"⚠ Could not save model: {ex}")

    # ── Predictions ────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba.ravel()

    # ── Inline evaluation ──────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, name: str = "Test Set") -> dict:
        print(f"\n{'=' * 60}")
        print(f"SVM EVALUATION — {name}")
        print(f"{'=' * 60}")

        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=['Rejected', 'Approved']))

        return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)


# ─── Model Evaluator ──────────────────────────────────────────────────────────
class ModelEvaluator:
    """Generates comprehensive evaluation plots for the SVM."""

    def __init__(self, model: SVMModel):
        self.model = model

    def evaluate(self, X, y_true, dataset_name: str = "Dataset"):
        print(f"\n{'=' * 70}")
        print(f"MODEL EVALUATION — {dataset_name}")
        print(f"{'=' * 70}")

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=['Rejected', 'Approved']))

        self._plot_evaluation(y_true, y_pred, y_prob, dataset_name)
        self._plot_prediction_analysis(y_true, y_pred, y_prob, dataset_name)

    # ── Main 2×2 evaluation plot ───────────────────────────────────────────
    def _plot_evaluation(self, y_true, y_pred, y_prob, name):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=DPI)

        # 1. Confusion Matrix
        cm      = confusion_matrix(y_true, y_pred)
        cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annots  = [[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)]
        sns.heatmap(cm, annot=np.array(annots), fmt='', cmap='Blues',
                    ax=axes[0, 0],
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'],
                    cbar_kws={'shrink': 0.8})
        axes[0, 0].set_title(f'Confusion Matrix — {name}', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val     = roc_auc_score(y_true, y_prob)
        axes[0, 1].plot(fpr, tpr, lw=2, label=f'ROC (AUC={auc_val:.3f})', color='#1f77b4')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
        axes[0, 1].fill_between(fpr, tpr, alpha=0.15, color='#1f77b4')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve — {name}', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Probability distribution by true label
        axes[1, 0].hist(y_prob[y_true == 0], bins=20, alpha=0.65,
                        label='Rejected (Actual)', color='#e74c3c', edgecolor='black')
        axes[1, 0].hist(y_prob[y_true == 1], bins=20, alpha=0.65,
                        label='Approved (Actual)', color='#2ecc71', edgecolor='black')
        axes[1, 0].set_xlabel('P(Approved)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Probability Distribution — {name}', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. Metrics bar chart
        m_names  = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        m_values = [accuracy_score(y_true, y_pred),
                    precision_score(y_true, y_pred, zero_division=0),
                    recall_score(y_true, y_pred, zero_division=0),
                    f1_score(y_true, y_pred, zero_division=0),
                    auc_val]
        bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = axes[1, 1].bar(m_names, m_values, color=bar_colors, edgecolor='black', alpha=0.75)
        axes[1, 1].set_ylim(0, 1.08)
        axes[1, 1].set_title(f'Performance Metrics — {name}', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, m_values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        fname = f'evaluation_{name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Evaluation plots saved → {fname}")

    # ── Prediction confidence + calibration ───────────────────────────────
    def _plot_prediction_analysis(self, y_true, y_pred, y_prob, name):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        correct   = y_prob[y_true == y_pred]
        incorrect = y_prob[y_true != y_pred]
        axes[0].hist(correct,   bins=20, alpha=0.65, label='Correct',
                     color='#2ecc71', edgecolor='black')
        axes[0].hist(incorrect, bins=20, alpha=0.65, label='Incorrect',
                     color='#e74c3c', edgecolor='black')
        axes[0].set_xlabel('Predicted Probability P(Approved)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Prediction Confidence — {name}', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        n_bins    = 10
        edges     = np.linspace(0, 1, n_bins + 1)
        m_probs, m_true = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() > 0:
                m_probs.append(y_prob[mask].mean())
                m_true.append(y_true[mask].mean())
        axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
        axes[1].plot(m_probs, m_true, 'o-', lw=2, ms=8, color='#1f77b4',
                     label='SVM calibration')
        axes[1].fill_between(m_probs, m_true, np.interp(m_probs, [0, 1], [0, 1]),
                             alpha=0.15, color='#1f77b4')
        axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Fraction of Positives')
        axes[1].set_title(f'Calibration Plot — {name}', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'predictions_{name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Prediction analysis saved → {fname}")


# ─── ML Pipeline ──────────────────────────────────────────────────────────────
class MLPipeline:
    """End-to-end SVM pipeline for Loan Approval Prediction."""

    def __init__(self):
        self.loader    = DatasetLoader(DATASET_PATH)
        self.processor = None
        self.model     = None
        self.evaluator = None

    def run(self):
        print(f"\n{'=' * 70}")
        print("SVM PIPELINE — LOAN APPROVAL PREDICTION")
        print(f"{'=' * 70}")

        # 1️⃣  Load
        data, target = self.loader.load_data()

        # 1.5️⃣  Visualise
        visualizer = LoanVisualizer(data, target)
        visualizer.visualize()

        # 2️⃣  Validate
        validator = DatasetValidator(data, target)
        validator.verify_dataset()

        # 3️⃣  Process
        self.processor = DatasetProcessor(data, target)
        proc_data, proc_target = self.processor.process_dataset()

        # 4️⃣  Split
        X_train, X_test, y_train, y_test = train_test_split(
            proc_data.values,
            proc_target.values,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=proc_target,
        )
        print(f"\n✓ Train: {X_train.shape}  |  Test: {X_test.shape}")

        # 5️⃣  Train (default hyperparams)
        self.model = SVMModel()
        self.model.fit(X_train, y_train)

        # 6️⃣  Evaluate
        self.evaluator = ModelEvaluator(self.model)
        self.evaluator.evaluate(X_train, y_train, "Training Set")
        self.evaluator.evaluate(X_test,  y_test,  "Test Set")

        # 7️⃣  Cross-validation
        self._perform_cross_validation(proc_data.values, proc_target.values)

        # 8️⃣  Hyperparameter tuning (GridSearchCV)
        best_model = self._tune_hyperparameters(X_train, y_train, X_test, y_test)

        # 9️⃣  Predict a new applicant
        self._test_new_applicant(X_train.shape[1], best_model)

    # ── 5-Fold Cross-Validation ────────────────────────────────────────────
    def _perform_cross_validation(self, X, y):
        print(f"\n{'=' * 70}")
        print("5-FOLD STRATIFIED CROSS-VALIDATION")
        print(f"{'=' * 70}")

        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale',
                  probability=True, random_state=RANDOM_STATE)

        metrics = {
            'accuracy':  cross_val_score(svm, X, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(svm, X, y, cv=cv, scoring='precision'),
            'recall':    cross_val_score(svm, X, y, cv=cv, scoring='recall'),
            'f1':        cross_val_score(svm, X, y, cv=cv, scoring='f1'),
            'roc_auc':   cross_val_score(svm, X, y, cv=cv, scoring='roc_auc'),
        }

        for name, scores in metrics.items():
            print(f"{name.capitalize():10s}: {scores.mean():.4f}  (+/- {scores.std():.4f})")

        stability = "GOOD generalisation ✓" if metrics['accuracy'].std() < 0.05 else "HIGH variance ⚠"
        print(f"\n✓ Cross-validation indicates {stability}")

        # ── Plot CV fold results ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
        fold_labels = [f'Fold {i+1}' for i in range(5)]
        x = np.arange(5)
        width = 0.15
        for i, (metric, scores) in enumerate(metrics.items()):
            ax.bar(x + i * width, scores, width, label=metric.capitalize(), alpha=0.8)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(fold_labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('5-Fold Cross-Validation Results', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("loan_cv_results.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] CV results plot saved")

    # ── GridSearchCV Tuning ────────────────────────────────────────────────
    def _tune_hyperparameters(self, X_train, y_train, X_test, y_test) -> SVMModel:
        print(f"\n{'=' * 70}")
        print("HYPERPARAMETER TUNING (GridSearchCV)")
        print(f"{'=' * 70}")
        print(f"Parameter grid: {PARAM_GRID}")

        base_svm = SVC(probability=True, random_state=RANDOM_STATE)
        cv       = StratifiedKFold(n_splits=TUNE_CV, shuffle=True, random_state=RANDOM_STATE)
        grid     = GridSearchCV(base_svm, PARAM_GRID, cv=cv,
                                scoring='roc_auc', n_jobs=TUNE_JOBS, verbose=1)
        grid.fit(X_train, y_train)

        print(f"\n✓ Best parameters : {grid.best_params_}")
        print(f"✓ Best CV ROC-AUC  : {grid.best_score_:.4f}")

        # Build optimised SVMModel
        bp = grid.best_params_
        best_svm        = SVMModel(kernel=bp['kernel'], C=bp['C'], gamma=bp['gamma'])
        best_svm.model  = grid.best_estimator_
        best_svm.is_fitted = True

        print("\n--- Tuned Model Evaluation ---")
        best_svm.evaluate(X_test, y_test, "Tuned Test Set")

        # ── GridSearchCV heatmap ───────────────────────────────────────────
        results = pd.DataFrame(grid.cv_results_)
        # Only use RBF results for the heatmap (linear has no meaningful gamma axis)
        rbf_mask = results['param_kernel'] == 'rbf'
        if rbf_mask.sum() > 0:
            pivot = results[rbf_mask].pivot_table(
                index='param_C', columns='param_gamma', values='mean_test_score')
            plt.figure(figsize=(8, 5), dpi=DPI)
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu')
            plt.title('GridSearchCV ROC-AUC (RBF kernel)', fontsize=12, fontweight='bold')
            plt.xlabel('Gamma')
            plt.ylabel('C')
            plt.tight_layout()
            plt.savefig("loan_gridsearch_heatmap.png", dpi=DPI, bbox_inches='tight')
            plt.close()
            print("[OK] GridSearchCV heatmap saved")

        return best_svm

    # ── Single Applicant Prediction ────────────────────────────────────────
    def _test_new_applicant(self, n_features: int, model: SVMModel):
        print(f"\n{'=' * 70}")
        print("NEW LOAN APPLICANT PREDICTION")
        print(f"{'=' * 70}")

        means  = self.processor.feature_means.values
        stds   = self.processor.feature_stds.values.copy()
        stds[stds == 0] = 1.0

        raw        = means + np.random.randn(n_features) * stds
        scaled     = ((raw - means) / stds).reshape(1, -1)

        prob = model.predict_proba(scaled)[0]
        pred = model.predict(scaled)[0]
        status = "✅ Approved" if pred == 1 else "❌ Rejected"

        print(f"Predicted P(Approved): {prob:.4f}")
        print(f"Decision             : {status}")


# ─── Entry Point ──────────────────────────────────────────────────────────────
def main():
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n❌ Pipeline error:")
        print(f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()