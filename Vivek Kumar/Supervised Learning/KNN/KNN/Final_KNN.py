import warnings 
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

#Global Variales
RANDOM_STATE = 42
DATASET_PATH = "diabetes.csv"  # Binary classification dataset
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# KNN Hyperparameters
K_NEIGHBORS = 5              # Number of nearest neighbors to vote
DISTANCE_METRIC = 'euclidean'  # 'euclidean' | 'manhattan' | 'minkowski'
MINKOWSKI_P = 2              # p=2 → Euclidean, p=1 → Manhattan (used when metric='minkowski')
WEIGHTS = 'uniform'          # 'uniform' (all neighbors vote equally) | 'distance' (closer = more weight)

# KNN Tuning
K_MIN = 1                    # Minimum K to search during tuning
K_MAX = 21                   # Maximum K to search during tuning (odd values preferred)
K_STEP = 2                   # Step size (use 2 to always stay odd → avoids ties)

# Model Persistence
MODEL_SAVE_PATH = "knn_model.pkl"

# Visualization Configuration
FIGURE_SIZE = (12, 8)
DPI = 100
STYLE = 'seaborn-v0_8-darkgrid'

@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def __str__(self) -> str:
        """Return formatted string representation of metrics."""
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
    
class DatasetLoader:
    """Class responsible for loading and initial dataset operations."""

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self.data: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.feature_names: Optional[list] = None

    def _generate_synthetic_dataset(self, n_samples: int = 768) -> Tuple[pd.DataFrame, pd.Series]:

        np.random.seed(RANDOM_STATE)

        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # Class 0 → Non-diabetic
        n_class_0 = n_samples // 2

        # Add noise to make data more realistic (overlap between classes)
        noise_scale = 1.2  # Increase overlap for realistic scenarios
        
        features_class_0 = {
            'Pregnancies': np.random.randint(0, 6, n_class_0),
            'Glucose': np.clip(np.random.normal(110, 25, n_class_0) + np.random.normal(0, noise_scale*5, n_class_0), 40, 250),
            'BloodPressure': np.clip(np.random.normal(70, 15, n_class_0) + np.random.normal(0, noise_scale*3, n_class_0), 20, 140),
            'SkinThickness': np.clip(np.random.normal(20, 12, n_class_0) + np.random.normal(0, noise_scale*2, n_class_0), 0, 100),
            'Insulin': np.clip(np.random.normal(80, 60, n_class_0) + np.random.normal(0, noise_scale*15, n_class_0), 0, 500),
            'BMI': np.clip(np.random.normal(28, 8, n_class_0) + np.random.normal(0, noise_scale*1.5, n_class_0), 18, 67),
            'DiabetesPedigreeFunction': np.clip(np.random.normal(0.4, 0.3, n_class_0) + np.random.normal(0, noise_scale*0.1, n_class_0), 0.08, 2.42),
            'Age': np.random.randint(21, 45, n_class_0),
        }

        # Class 1 → Diabetic
        n_class_1 = n_samples - n_class_0

        features_class_1 = {
            'Pregnancies': np.random.randint(2, 10, n_class_1),
            'Glucose': np.clip(np.random.normal(150, 35, n_class_1) + np.random.normal(0, noise_scale*8, n_class_1), 40, 250),
            'BloodPressure': np.clip(np.random.normal(80, 18, n_class_1) + np.random.normal(0, noise_scale*4, n_class_1), 20, 140),
            'SkinThickness': np.clip(np.random.normal(30, 14, n_class_1) + np.random.normal(0, noise_scale*3, n_class_1), 0, 100),
            'Insulin': np.clip(np.random.normal(150, 80, n_class_1) + np.random.normal(0, noise_scale*20, n_class_1), 0, 500),
            'BMI': np.clip(np.random.normal(35, 10, n_class_1) + np.random.normal(0, noise_scale*2, n_class_1), 18, 67),
            'DiabetesPedigreeFunction': np.clip(np.random.normal(0.8, 0.4, n_class_1) + np.random.normal(0, noise_scale*0.15, n_class_1), 0.08, 2.42),
            'Age': np.random.randint(30, 65, n_class_1),
        }

        data_dict = {}
        for feature in self.feature_names:
            data_dict[feature] = np.concatenate([
                features_class_0[feature],
                features_class_1[feature]
            ])

        self.data = pd.DataFrame(data_dict)
        self.target = pd.Series(
            np.concatenate([np.zeros(n_class_0), np.ones(n_class_1)]),
            name='Outcome'
        )

        indices = np.random.permutation(n_samples)
        self.data = self.data.iloc[indices].reset_index(drop=True)
        self.target = self.target.iloc[indices].reset_index(drop=True)

        return self.data, self.target

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        # If a CSV dataset exists at the provided path, load it; otherwise synthesize data
        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                if 'Outcome' in df.columns:
                    self.data = df.drop(columns=['Outcome'])
                    self.target = df['Outcome']
                else:
                    # Assume last column is target if not named
                    self.data = df.iloc[:, :-1]
                    self.target = df.iloc[:, -1]

                self.feature_names = list(self.data.columns)
                print(f"✓ Loaded dataset from: {self.dataset_path}")
                print(f"✓ Samples: {len(self.data)}")
                print(f"✓ Features: {len(self.feature_names)}")
                return self.data, self.target
            except Exception as ex:
                print(f"⚠ Failed to load {self.dataset_path}: {ex}")
                print("⚠ Falling back to synthetic dataset generation")

        self.data, self.target = self._generate_synthetic_dataset()

        print(f"✓ Synthetic diabetes dataset generated")
        print(f"✓ Samples: {len(self.data)}")
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ Feature names: {', '.join(self.feature_names)}")
        print(f"✓ Classes: 0 (Non-Diabetic), 1 (Diabetic)")

        return self.data, self.target
    
class DatasetValidator:
    """Class responsible for dataset validation and verification."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data
        self.target = target

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        validation_passed = True

        if self.data.empty or self.target.empty:
            print("✗ ERROR: Dataset is empty!")
            return False
        print("✓ Dataset is not empty")

        print(f"\n--- Dataset Shape ---")
        print(f"Features shape: {self.data.shape}")
        print(f"Target shape: {self.target.shape}")

        if self.data.shape[0] != self.target.shape[0]:
            print("✗ ERROR: Features and target have different number of rows!")
            return False
        print("✓ Features and target have matching rows")

        print(f"\n--- Missing Values ---")
        missing_features = self.data.isnull().sum().sum()
        missing_target = self.target.isnull().sum()
        print(f"Missing values in features: {missing_features}")
        print(f"Missing values in target: {missing_target}")

        if missing_features > 0 or missing_target > 0:
            print("⚠ WARNING: Dataset contains missing values")
            validation_passed = False
        else:
            print("✓ No missing values detected")

        print(f"\n--- Data Types ---")
        print(self.data.dtypes)

        non_numeric = self.data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"⚠ WARNING: Non-numeric columns detected: {list(non_numeric)}")
            validation_passed = False
        else:
            print("✓ All features are numeric")

        print(f"\n--- First 5 Rows ---")
        print(self.data.head())

        print(f"\n--- Statistical Summary ---")
        print(self.data.describe())

        print(f"\n--- Class Distribution ---")
        class_counts = self.target.value_counts()
        print(class_counts)
        print(f"Class balance: {class_counts.min() / class_counts.max():.2f}")

        inf_count = np.isinf(self.data.values).sum()
        if inf_count > 0:
            print(f"⚠ WARNING: {inf_count} infinite values detected")
            validation_passed = False
        else:
            print("✓ No infinite values detected")

        return validation_passed

class DatasetProcessor:
    """Class responsible for dataset processing and transformation."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data.copy()
        self.target = target.copy()
        self.processed_data: Optional[pd.DataFrame] = None
        self.processed_target: Optional[pd.Series] = None

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        # Handle missing values
        print("\n--- Handling Missing Values ---")
        missing_before = self.data.isnull().sum().sum()
        self.data = self.data.fillna(self.data.mean())
        self.target = self.target.fillna(self.target.mode()[0])
        missing_after = self.data.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")
        print(f"Missing values after:  {missing_after}")

        # ─── KNN-SPECIFIC NOTE ────────────────────────────────────────────
        # Standardization is CRITICAL for KNN. Without it, features with
        # large ranges (e.g. area_mean ~500) completely dominate distance
        # calculations over features with small ranges (e.g. smoothness ~0.09).
        # Euclidean distance treats every dimension equally ONLY after scaling.
        # ──────────────────────────────────────────────────────────────────
        print("\n--- Feature Standardization (Z-Score) ---")
        print("  ⓘ  Standardization is mandatory for KNN — distance metrics")
        print("     are scale-sensitive. Without it, large-range features")
        print("     dominate neighbor calculations.")
        self.feature_means = self.data.mean()
        self.feature_stds = self.data.std()
        self.processed_data = (self.data - self.feature_means) / self.feature_stds
        print("✓ Features standardized (mean=0, std=1)")

        # Ensure target is binary
        print("\n--- Target Conversion ---")
        unique_values = self.target.unique()
        if len(unique_values) > 2:
            print("⚠ WARNING: More than 2 classes detected. Converting to binary.")
            self.processed_target = (self.target > self.target.median()).astype(int)
        else:
            self.processed_target = self.target.astype(int)
        print(f"Target classes: {sorted(self.processed_target.unique())}")

        print(f"\n--- Processed Dataset Shape ---")
        print(f"Processed features: {self.processed_data.shape}")
        print(f"Processed target:   {self.processed_target.shape}")

        return self.processed_data, self.processed_target

class DiabetesVisualizer:
    """Visualization class for Diabetes dataset."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data
        self.target = target
        plt.style.use(STYLE)

    # =====================================================
    def visualize(self):
        print("\n" + "=" * 70)
        print("DIABETES DATASET VISUALIZATION")
        print("=" * 70)

        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_feature_distributions()
        self.plot_feature_boxplots()
        self.plot_top_correlations()
        self.plot_feature_violin_plots()
        self.plot_feature_kde_plots()
        self.plot_feature_statistics()
        self.plot_feature_importance()
        self.plot_3d_scatter()

        print("✓ All diabetes visualizations saved")

    # =====================================================
    # 1️⃣ CLASS DISTRIBUTION
    # =====================================================
    def plot_target_distribution(self):
        counts = self.target.value_counts().sort_index()

        plt.figure(figsize=(14, 5), dpi=DPI)

        # Bar chart
        plt.subplot(1, 2, 1)
        bars = plt.bar(['No Diabetes (0)', 'Diabetes (1)'], counts.values,
                       color=['green', 'red'], edgecolor='black', alpha=0.7)
        plt.title("Diabetes Class Distribution", fontsize=12, fontweight='bold')
        plt.ylabel("Count", fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts.values,
                labels=['No Diabetes', 'Diabetes'],
                autopct='%1.1f%%',
                colors=['green', 'red'],
                startangle=90,
                explode=(0.05, 0.05))
        plt.title("Class Ratio", fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig("diabetes_target_distribution.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Target distribution plot saved")

    # =====================================================
    # 2️⃣ CORRELATION HEATMAP
    # =====================================================
    def plot_correlation_heatmap(self):
        df = self.data.copy()
        df["Outcome"] = self.target

        plt.figure(figsize=(14, 10), dpi=DPI)
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": 0.8})
        plt.title("Diabetes Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("diabetes_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Correlation heatmap saved")

    # =====================================================
    # 3️⃣ FEATURE DISTRIBUTIONS
    # =====================================================
    def plot_feature_distributions(self):
        num_features = len(self.data.columns)
        cols = 3
        rows = (num_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(self.data.columns):
            axes[i].hist(self.data[self.target == 0][col], bins=30,
                         alpha=0.6, label='No Diabetes', color='green', edgecolor='black')
            axes[i].hist(self.data[self.target == 1][col], bins=30,
                         alpha=0.6, label='Diabetes', color='red', edgecolor='black')
            axes[i].set_title(col, fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("diabetes_feature_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature distributions plot saved")

    # =====================================================
    # 4️⃣ BOXPLOTS (MOST IMPORTANT FEATURES)
    # =====================================================
    def plot_feature_boxplots(self):
        corr = self.data.corrwith(self.target).abs().sort_values(ascending=False)
        top_features = corr.head(4).index

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        for i, feature in enumerate(top_features):
            sns.boxplot(x=self.target, y=self.data[feature], ax=axes[i],
                       palette=['green', 'red'])
            axes[i].set_title(f"{feature} vs Outcome (Corr: {corr[feature]:.3f})",
                             fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Outcome')
            axes[i].set_ylabel(feature)
            axes[i].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("diabetes_feature_boxplots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature boxplots saved")

    # =====================================================
    # 5️⃣ TOP CORRELATIONS SCATTER PLOTS
    # =====================================================
    def plot_top_correlations(self):
        """Plot scatter plots for top correlated features with target."""
        correlations = self.data.corrwith(self.target).abs().sort_values(ascending=False)
        top_features = correlations.head(4).index.tolist()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        for idx, feature in enumerate(top_features):
            # Separate by class
            class_0 = self.data[self.target == 0]
            class_1 = self.data[self.target == 1]
            
            axes[idx].scatter(class_0[feature], [0]*len(class_0), alpha=0.6, s=50,
                            label='No Diabetes', color='green', edgecolors='darkgreen')
            axes[idx].scatter(class_1[feature], [1]*len(class_1), alpha=0.6, s=50,
                            label='Diabetes', color='red', edgecolors='darkred')
            axes[idx].set_xlabel(feature, fontsize=11)
            axes[idx].set_ylabel('Class')
            axes[idx].set_yticks([0, 1])
            axes[idx].set_yticklabels(['No Diabetes', 'Diabetes'])
            axes[idx].set_title(f'{feature} vs Outcome (Corr: {correlations[feature]:.3f})',
                              fontsize=11, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("diabetes_top_correlations.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Top correlations scatter plot saved")

    # =====================================================
    # 6️⃣ FEATURE VIOLIN PLOTS
    # =====================================================
    def plot_feature_violin_plots(self):
        """Plot violin plots for all features by class."""
        num_features = len(self.data.columns)
        cols = 3
        rows = (num_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4), dpi=DPI)
        axes = axes.flatten()
        
        # Prepare data for violin plot
        plot_data = self.data.copy()
        plot_data['Outcome'] = self.target
        
        for i, col in enumerate(self.data.columns):
            sns.violinplot(data=plot_data, x='Outcome', y=col, ax=axes[i],
                          palette=['green', 'red'], alpha=0.7)
            axes[i].set_title(f"{col} Distribution by Class", fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Value')
            axes[i].set_xticklabels(['No Diabetes', 'Diabetes'])
            axes[i].grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(num_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig("diabetes_feature_violin_plots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature violin plots saved")

    # =====================================================
    # 7️⃣ FEATURE KDE PLOTS
    # =====================================================
    def plot_feature_kde_plots(self):
        """Plot KDE density plots for top features."""
        correlations = self.data.corrwith(self.target).abs().sort_values(ascending=False)
        top_features = correlations.head(8).index.tolist()
        
        cols = 2
        rows = (len(top_features) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4), dpi=DPI)
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            class_0_data = self.data[self.target == 0][feature]
            class_1_data = self.data[self.target == 1][feature]
            
            class_0_data.plot.kde(ax=axes[idx], linewidth=2, label='No Diabetes', 
                                 color='green', alpha=0.7)
            class_1_data.plot.kde(ax=axes[idx], linewidth=2, label='Diabetes', 
                                 color='red', alpha=0.7)
            
            axes[idx].fill_between(class_0_data.plot.kde(ax=axes[idx]).get_lines()[0].get_xdata(),
                                  0, alpha=0.1, color='green')
            axes[idx].set_title(f"{feature} - KDE Density Plot", fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig("diabetes_feature_kde_plots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature KDE plots saved")

    # =====================================================
    # 8️⃣ FEATURE STATISTICS HEATMAP
    # =====================================================
    def plot_feature_statistics(self):
        """Plot feature statistics (mean, std) by class as heatmap."""
        stats_by_class = pd.DataFrame()
        
        for feature in self.data.columns:
            class_0_mean = self.data[self.target == 0][feature].mean()
            class_0_std = self.data[self.target == 0][feature].std()
            class_1_mean = self.data[self.target == 1][feature].mean()
            class_1_std = self.data[self.target == 1][feature].std()
            
            stats_by_class[f"{feature}\n(Mean)"] = [class_0_mean, class_1_mean]
            stats_by_class[f"{feature}\n(Std)"] = [class_0_std, class_1_std]
        
        stats_by_class.index = ['No Diabetes', 'Diabetes']
        
        # Normalize for better heatmap visualization
        stats_normalized = (stats_by_class - stats_by_class.min().min()) / \
                          (stats_by_class.max().max() - stats_by_class.min().min())
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=DPI)
        
        # Raw statistics
        sns.heatmap(stats_by_class, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=axes[0], cbar_kws={"shrink": 0.8})
        axes[0].set_title('Feature Statistics by Class (Raw Values)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Class')
        
        # Normalized statistics
        sns.heatmap(stats_normalized, annot=stats_by_class.values, fmt='.2f', cmap='RdYlGn', 
                   ax=axes[1], cbar_kws={"shrink": 0.8})
        axes[1].set_title('Feature Statistics by Class (Normalized)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Class')
        
        plt.tight_layout()
        plt.savefig("diabetes_feature_statistics.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature statistics heatmap saved")

    # =====================================================
    # 9️⃣ FEATURE IMPORTANCE
    # =====================================================
    def plot_feature_importance(self):
        """Plot feature importance based on correlation and variance."""
        # Correlation with target
        correlations = self.data.corrwith(self.target).abs().sort_values(ascending=False)
        
        # Variance by class difference
        class_0_var = self.data[self.target == 0].var()
        class_1_var = self.data[self.target == 1].var()
        variance_importance = (class_0_var + class_1_var).sort_values(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=DPI)
        
        # Correlation importance
        colors_corr = ['#d62728' if x == correlations.max() else '#1f77b4' 
                      for x in correlations.values]
        axes[0].barh(correlations.index, correlations.values, color=colors_corr, 
                    edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Absolute Correlation with Target', fontsize=11)
        axes[0].set_title('Feature Importance - Correlation Analysis', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(correlations.values):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        # Variance importance
        colors_var = ['#d62728' if x == variance_importance.max() else '#2ca02c' 
                     for x in variance_importance.values]
        axes[1].barh(variance_importance.index, variance_importance.values, color=colors_var,
                    edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Variance (Sum of Classes)', fontsize=11)
        axes[1].set_title('Feature Importance - Variance Analysis', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(variance_importance.values):
            axes[1].text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig("diabetes_feature_importance.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature importance plot saved")

    # =====================================================
    # 🔟 3D SCATTER PLOT
    # =====================================================
    def plot_3d_scatter(self):
        """Plot 3D scatter plot for top 3 features."""
        from mpl_toolkits.mplot3d import Axes3D
        
        correlations = self.data.corrwith(self.target).abs().sort_values(ascending=False)
        top_3_features = correlations.head(3).index.tolist()
        
        if len(top_3_features) < 3:
            print("[OK] Skipping 3D plot (fewer than 3 features)")
            return
        
        fig = plt.figure(figsize=(12, 9), dpi=DPI)
        ax = fig.add_subplot(111, projection='3d')
        
        class_0_indices = self.target == 0
        class_1_indices = self.target == 1
        
        # Plot class 0
        ax.scatter(self.data[class_0_indices][top_3_features[0]],
                  self.data[class_0_indices][top_3_features[1]],
                  self.data[class_0_indices][top_3_features[2]],
                  c='green', label='No Diabetes', s=30, alpha=0.6, edgecolors='darkgreen')
        
        # Plot class 1
        ax.scatter(self.data[class_1_indices][top_3_features[0]],
                  self.data[class_1_indices][top_3_features[1]],
                  self.data[class_1_indices][top_3_features[2]],
                  c='red', label='Diabetes', s=30, alpha=0.6, edgecolors='darkred')
        
        ax.set_xlabel(f'{top_3_features[0]}', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{top_3_features[1]}', fontsize=11, fontweight='bold')
        ax.set_zlabel(f'{top_3_features[2]}', fontsize=11, fontweight='bold')
        ax.set_title('3D Visualization of Top 3 Features', fontsize=13, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("diabetes_3d_scatter.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 3D scatter plot saved")

class KNNModel:
    """
    K-Nearest Neighbors model for Diabetes Prediction
    """

    def __init__(self, k: int = 5, metric: str = 'euclidean', weights: str = 'uniform', p: Optional[int] = None):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.p = p
        params = {"n_neighbors": self.k, "metric": self.metric, "weights": self.weights}
        if self.metric == 'minkowski' and self.p is not None:
            params['p'] = self.p
        self.model = KNeighborsClassifier(**params)

    # =====================================================
    # TRAINING
    # =====================================================
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print("\n" + "=" * 60)
        print("TRAINING KNN MODEL (DIABETES)")
        print("=" * 60)
        print(f"k (neighbors): {self.k}")

        self.model.fit(X_train, y_train)
        print("✓ KNN training completed")

        # Attempt to persist the trained model
        try:
            joblib.dump(self.model, MODEL_SAVE_PATH)
            print(f"✓ Model saved to {MODEL_SAVE_PATH}")
        except Exception as ex:
            print(f"⚠ Could not save model: {ex}")

    # =====================================================
    # PREDICTIONS
    # =====================================================
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X_test)
        # return probability for positive class when available
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    # =====================================================
    # EVALUATION
    # =====================================================
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, name="Test Set"):
        print("\n" + "=" * 60)
        print(f"KNN MODEL EVALUATION — {name}")
        print("=" * 60)

        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['No Diabetes', 'Diabetes']))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }
    
class ModelEvaluator:
    """Evaluates model performance for Diabetes prediction."""

    def __init__(self, model: KNNModel):
        self.model = model

    def evaluate(self, X, y_true, dataset_name="Dataset"):
        print(f"\n{'=' * 70}")
        print(f"MODEL EVALUATION — {dataset_name}")
        print(f"{'=' * 70}")

        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)

        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_true, y_pred_proba)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=['Non-Diabetic', 'Diabetic']
        ))

        # Create comprehensive evaluation plots
        self._plot_evaluation(y_true, y_pred, y_pred_proba, dataset_name)
        self._plot_prediction_distribution(y_true, y_pred, y_pred_proba, dataset_name)

    def _plot_evaluation(self, y_true, y_pred, y_pred_proba, dataset_name):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=DPI)

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm, annot=np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                                         for j in range(cm.shape[1])] 
                                        for i in range(cm.shape[0])]),
                    fmt='', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Non-Diabetic', 'Diabetic'],
                    yticklabels=['Non-Diabetic', 'Diabetic'],
                    cbar_kws={"shrink": 0.8})
        axes[0, 0].set_title(f'Confusion Matrix - {dataset_name}', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_val = roc_auc_score(y_true, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={auc_val:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        axes[0, 1].fill_between(fpr, tpr, alpha=0.2)
        axes[0, 1].set_xlabel('False Positive Rate', fontsize=11)
        axes[0, 1].set_ylabel('True Positive Rate', fontsize=11)
        axes[0, 1].set_title(f'ROC Curve - {dataset_name}', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Prediction Probability Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=20, alpha=0.6, 
                       label='No Diabetes (Actual)', color='green', edgecolor='black')
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=20, alpha=0.6, 
                       label='Diabetes (Actual)', color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Predicted Probability (Diabetic)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title(f'Prediction Probability Distribution - {dataset_name}', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. Classification Metrics Bar Chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_values = [accuracy_score(y_true, y_pred),
                         precision_score(y_true, y_pred, zero_division=0),
                         recall_score(y_true, y_pred, zero_division=0),
                         f1_score(y_true, y_pred, zero_division=0),
                         auc_val]
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors_bar, 
                             edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title(f'Performance Metrics - {dataset_name}', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylim([0, 1.05])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for bar, val in zip(bars, metrics_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = f'evaluation_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Evaluation plots saved for {dataset_name}")

    def _plot_prediction_distribution(self, y_true, y_pred, y_pred_proba, dataset_name):
        """Plot prediction analysis and class-wise distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        # Classification Correctness
        is_correct = (y_true == y_pred)
        correct_probs = y_pred_proba[is_correct]
        incorrect_probs = y_pred_proba[~is_correct]

        axes[0].hist(correct_probs, bins=20, alpha=0.6, label='Correct Predictions', 
                    color='green', edgecolor='black')
        axes[0].hist(incorrect_probs, bins=20, alpha=0.6, label='Incorrect Predictions', 
                    color='red', edgecolor='black')
        axes[0].set_xlabel('Predicted Probability', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title(f'Prediction Confidence Analysis - {dataset_name}', 
                         fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Calibration Plot (Reliability Diagram)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        mean_probs = []
        mean_true = []
        
        for i in range(n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if mask.sum() > 0:
                mean_probs.append(y_pred_proba[mask].mean())
                mean_true.append(y_true[mask].mean())
            else:
                mean_probs.append(bin_centers[i])
                mean_true.append(0)

        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        axes[1].plot(mean_probs, mean_true, 'o-', linewidth=2, markersize=8, 
                    label='Model Calibration', color='#1f77b4')
        axes[1].fill_between(mean_probs, mean_true, np.linspace(0, 1, len(mean_probs)), 
                            alpha=0.2, color='#1f77b4')
        axes[1].set_xlabel('Mean Predicted Probability', fontsize=11)
        axes[1].set_ylabel('Actual Positive Rate', fontsize=11)
        axes[1].set_title(f'Calibration Plot - {dataset_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'predictions_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Prediction distribution plots saved for {dataset_name}")

class MLPipeline:
    """Main pipeline for Diabetes Prediction using KNN."""

    def __init__(self):
        self.loader = DatasetLoader(DATASET_PATH)
        self.processor = None
        self.model = None
        self.evaluator = None

    def run(self):
        print("\n" + "=" * 70)
        print("KNN PIPELINE — DIABETES PREDICTION")
        print("=" * 70)

        # 1️⃣ Load Data
        data, target = self.loader.load_data()

        # 1.5️⃣ Visualize Dataset
        visualizer = DiabetesVisualizer(data, target)
        visualizer.visualize()

        # 2️⃣ Process
        self.processor = DatasetProcessor(data, target)
        processed_data, processed_target = self.processor.process_dataset()

        # 3️⃣ Split
        X_train, X_test, y_train, y_test = train_test_split(
            processed_data.values,
            processed_target.values,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=processed_target
        )

        # 4️⃣ Train Model
        self.model = KNNModel(k=5)
        self.model.fit(X_train, y_train)

        # 5️⃣ Evaluate
        self.evaluator = ModelEvaluator(self.model)
        self.evaluator.evaluate(X_train, y_train, "Training Set")
        self.evaluator.evaluate(X_test, y_test, "Test Set")
        
        # 5.5️⃣ K-Fold Cross-Validation
        self._perform_cross_validation(processed_data.values, processed_target.values)

        # 6️⃣ Test New Patient
        self._test_new_patient(X_train.shape[1])

    def _test_new_patient(self, n_features):
        print(f"\n{'=' * 70}")
        print("NEW PATIENT PREDICTION")
        print(f"{'=' * 70}")

        # Create a realistic new patient in the original feature space using
        # the processor's means and stds, then standardize for prediction.
        means = self.processor.feature_means.values
        stds = self.processor.feature_stds.values.copy()
        stds[stds == 0] = 1.0

        raw_patient = means + np.random.randn(n_features) * stds
        new_patient = ((raw_patient - means) / stds).reshape(1, -1)

        prob = self.model.predict_proba(new_patient)[0]
        pred = self.model.predict(new_patient)[0]

        status = "Diabetic" if pred == 1 else "Non-Diabetic"
        print(f"Predicted Probability (Diabetic): {prob:.4f}")
        print(f"Prediction: {status}")
    
    def _perform_cross_validation(self, X, y):
        """Perform k-fold cross-validation for robust evaluation."""
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        print(f"\n{'=' * 70}")
        print("K-FOLD CROSS-VALIDATION (5-Fold)")
        print(f"{'=' * 70}")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        knn_cv = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
        
        # Compute cross-validation scores
        accuracy_scores = cross_val_score(knn_cv, X, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(knn_cv, X, y, cv=cv, scoring='precision')
        recall_scores = cross_val_score(knn_cv, X, y, cv=cv, scoring='recall')
        f1_scores = cross_val_score(knn_cv, X, y, cv=cv, scoring='f1')
        roc_auc_scores = cross_val_score(knn_cv, X, y, cv=cv, scoring='roc_auc')
        
        print(f"\nAccuracy:  {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
        print(f"Precision: {precision_scores.mean():.4f} (+/- {precision_scores.std():.4f})")
        print(f"Recall:    {recall_scores.mean():.4f} (+/- {recall_scores.std():.4f})")
        print(f"F1-Score:  {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        print(f"ROC-AUC:   {roc_auc_scores.mean():.4f} (+/- {roc_auc_scores.std():.4f})")
        
        print(f"\n✓ Cross-validation indicates {'GOOD generalization ✓' if accuracy_scores.std() < 0.05 else 'HIGH variance ⚠'}")

def main():
    """Main function to run the KNN ML pipeline."""
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n❌ ERROR during pipeline execution:")
        print(f"{type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    main()


    


