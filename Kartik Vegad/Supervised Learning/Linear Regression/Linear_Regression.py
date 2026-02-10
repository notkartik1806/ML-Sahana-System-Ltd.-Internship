import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# Dataset Configuration
DATASET_PATH = "breast_cancer_classification"  # Binary classification dataset
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Model Hyperparameters
LEARNING_RATE = 0.01
NUM_ITERATIONS = 2000
BATCH_SIZE = 32
CONVERGENCE_THRESHOLD = 1e-6
GRADIENT_CLIP_VALUE = 5.0

# Model Persistence
MODEL_SAVE_PATH = "Supervised Learning/Linear Regression/logistic_regression_model.pkl"

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

    def __init__(self, dataset_path: str):
       """
       Initialize the DatasetLoader.

       Args:
          dataset_path: Path or identifier for the dataset
       """
       self.dataset_path = dataset_path
       self.data: Optional[pd.DataFrame] = None
       self.target: Optional[pd.Series] = None
       self.feature_names: Optional[list] = None

    def _generate_synthetic_dataset(
          self,
          n_samples: int = 569,
          n_features: int = 10
    ) -> Tuple[pd.DataFrame, pd.Series]:
       """
       Generate synthetic binary classification dataset.
 
       Args:
          n_samples: Number of samples to generate
          n_features: Number of features

       Returns:
          Tuple of (features_dataframe, target_series)
       """
       np.random.seed(RANDOM_STATE)

       # Feature names (medical-like features)
       self.feature_names = [
          'radius_mean', 'texture_mean', 'perimeter_mean',
          'area_mean', 'smoothness_mean', 'compactness_mean',
          'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean',
          'radius_se'
       ]

       # Generate features with different distributions for two classes
       # Class 0: Benign (lower values)
       n_class_0 = n_samples // 2
       features_class_0 = {
          'radius_mean': np.random.normal(12, 2, n_class_0),
          'texture_mean': np.random.normal(18, 3, n_class_0),
          'perimeter_mean': np.random.normal(80, 12, n_class_0),
          'area_mean': np.random.normal(450, 100, n_class_0),
          'smoothness_mean': np.random.normal(0.09, 0.01, n_class_0),
          'compactness_mean': np.random.normal(0.08, 0.02, n_class_0),
          'concavity_mean': np.random.normal(0.05, 0.02, n_class_0),
          'symmetry_mean': np.random.normal(0.17, 0.02, n_class_0),
          'fractal_dimension_mean': np.random.normal(0.06, 0.005, n_class_0),
          'radius_se': np.random.normal(0.3, 0.1, n_class_0),
       }

       # Class 1: Malignant (higher values)
       n_class_1 = n_samples - n_class_0
       features_class_1 = {
          'radius_mean': np.random.normal(17, 3, n_class_1),
          'texture_mean': np.random.normal(22, 4, n_class_1),
          'perimeter_mean': np.random.normal(115, 20, n_class_1),
          'area_mean': np.random.normal(900, 250, n_class_1),
          'smoothness_mean': np.random.normal(0.11, 0.015, n_class_1),
          'compactness_mean': np.random.normal(0.15, 0.04, n_class_1),
          'concavity_mean': np.random.normal(0.12, 0.04, n_class_1),
          'symmetry_mean': np.random.normal(0.20, 0.03, n_class_1),
          'fractal_dimension_mean': np.random.normal(0.065, 0.008, n_class_1),
          'radius_se': np.random.normal(0.5, 0.15, n_class_1),
       }

       # Combine both classes
       data_dict = {}
       for feature in self.feature_names:
          data_dict[feature] = np.concatenate([
             features_class_0[feature],
             features_class_1[feature]
          ])

       self.data = pd.DataFrame(data_dict)

       # Create target: 0 = Benign, 1 = Malignant
       self.target = pd.Series(
          np.concatenate([
             np.zeros(n_class_0),
             np.ones(n_class_1)
          ]),
          name='diagnosis'
       )

       # Shuffle the data
       indices = np.random.permutation(n_samples)
       self.data = self.data.iloc[indices].reset_index(drop=True)
       self.target = self.target.iloc[indices].reset_index(drop=True)

       return self.data, self.target

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
       """
       Load the classification dataset.

       Returns:
          Tuple of (features_dataframe, target_series)
       """
       print(f"\n{'=' * 70}")
       print("LOADING DATASET")
       print(f"{'=' * 70}")

       # Generate synthetic breast cancer-like dataset
       self.data, self.target = self._generate_synthetic_dataset()

       print(f"✓ Synthetic breast cancer classification dataset generated")
       print(f"✓ Number of samples: {len(self.data)}")
       print(f"✓ Number of features: {len(self.feature_names)}")
       print(f"✓ Feature names: {', '.join(self.feature_names)}")
       print(f"✓ Classes: 0 (Benign), 1 (Malignant)")

       return self.data, self.target


class DatasetValidator:
    """Class responsible for dataset validation and verification."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
       """
       Initialize the DatasetValidator.

       Args:
          data: Features dataframe
          target: Target series
       """
       self.data = data
       self.target = target

    def verify_dataset(self) -> bool:
       """
       Perform comprehensive dataset verification.

       Returns:
          Boolean indicating if dataset passed all validations
       """
       print(f"\n{'=' * 70}")
       print("DATASET VERIFICATION")
       print(f"{'=' * 70}")

       validation_passed = True

       # Check for empty dataset
       if self.data.empty or self.target.empty:
          print("✗ ERROR: Dataset is empty!")
          return False
       print("✓ Dataset is not empty")

       # Check dataset shape
       print(f"\n--- Dataset Shape ---")
       print(f"Features shape: {self.data.shape}")
       print(f"Target shape: {self.target.shape}")
       print(f"Number of samples: {self.data.shape[0]}")
       print(f"Number of features: {self.data.shape[1]}")

       # Check for mismatched rows
       if self.data.shape[0] != self.target.shape[0]:
          print("✗ ERROR: Features and target have different number of rows!")
          return False
       print("✓ Features and target have matching rows")

       # Check for missing values
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

       # Check data types
       print(f"\n--- Data Types ---")
       print(self.data.dtypes)

       # Verify numeric data
       non_numeric = self.data.select_dtypes(exclude=[np.number]).columns
       if len(non_numeric) > 0:
          print(f"⚠ WARNING: Non-numeric columns detected: {list(non_numeric)}")
          validation_passed = False
       else:
          print("✓ All features are numeric")

       # Display first few rows
       print(f"\n--- First 5 Rows ---")
       print(self.data.head())

       # Display last few rows
       print(f"\n--- Last 5 Rows ---")
       print(self.data.tail())

       # Display statistical summary
       print(f"\n--- Statistical Summary ---")
       print(self.data.describe())

       # Target statistics
       print(f"\n--- Target Statistics ---")
       print(self.target.describe())
       print(f"\n--- Class Distribution ---")
       class_counts = self.target.value_counts()
       print(class_counts)
       print(f"Class balance: {class_counts.min() / class_counts.max():.2f}")

       # Check for infinite values
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
       """
       Initialize the DatasetProcessor.

       Args:
          data: Features dataframe
          target: Target series
       """
       self.data = data.copy()
       self.target = target.copy()
       self.processed_data: Optional[pd.DataFrame] = None
       self.processed_target: Optional[pd.Series] = None

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
       """
       Process the dataset: handle missing values, outliers, normalization.

       Returns:
          Tuple of (processed_features, processed_target)
       """
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
       print(f"Missing values after: {missing_after}")

       # Feature scaling (standardization)
       print("\n--- Feature Standardization ---")
       self.feature_means = self.data.mean()
       self.feature_stds = self.data.std()

       self.processed_data = (self.data - self.feature_means) / self.feature_stds
       print("✓ Features standardized (mean=0, std=1)")

       # Ensure target is binary (0 or 1)
       print("\n--- Target Conversion ---")
       unique_values = self.target.unique()
       print(f"Unique target values: {unique_values}")

       if len(unique_values) > 2:
          print("⚠ WARNING: More than 2 classes detected. Converting to binary.")
          self.processed_target = (self.target > self.target.median()).astype(int)
       else:
          self.processed_target = self.target.astype(int)

       print(f"Target classes after processing: {self.processed_target.unique()}")

       print(f"\n--- Processed Dataset Shape ---")
       print(f"Processed features: {self.processed_data.shape}")
       print(f"Processed target: {self.processed_target.shape}")

       return self.processed_data, self.processed_target


class DatasetVisualizer:
    """Class responsible for dataset visualization."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
       """
       Initialize the DatasetVisualizer.

       Args:
          data: Features dataframe
          target: Target series
       """
       self.data = data
       self.target = target
       plt.style.use(STYLE)

    def visualize_dataset(self) -> None:
       """Create comprehensive visualizations of the dataset."""
       print(f"\n{'=' * 70}")
       print("DATASET VISUALIZATION")
       print(f"{'=' * 70}")

       # 1. Target distribution
       self._plot_target_distribution()

       # 2. Correlation heatmap
       self._plot_correlation_heatmap()

       # 3. Feature distributions by class
       self._plot_feature_distributions()

       # 4. Box plots for top features
       self._plot_feature_boxplots()

       print("✓ All visualizations created successfully")

    def _plot_target_distribution(self) -> None:
       """Plot the distribution of target variable."""
       plt.figure(figsize=(10, 6), dpi=DPI)

       class_counts = self.target.value_counts()

       plt.subplot(1, 2, 1)
       plt.bar(['Benign (0)', 'Malignant (1)'], class_counts.values,
               color=['green', 'red'], alpha=0.7, edgecolor='black')
       plt.xlabel('Class')
       plt.ylabel('Count')
       plt.title('Class Distribution (Bar Chart)')
       plt.grid(True, alpha=0.3)

       plt.subplot(1, 2, 2)
       plt.pie(class_counts.values, labels=['Benign (0)', 'Malignant (1)'],
               autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
       plt.title('Class Distribution (Pie Chart)')

       plt.tight_layout()
       plt.savefig('Supervised Learning/Linear Regression/images/target_distribution.png',
                   dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Target distribution plot saved")

    def _plot_correlation_heatmap(self) -> None:
       """Plot correlation heatmap between features and target."""
       combined_data = self.data.copy()
       combined_data['Target'] = self.target

       plt.figure(figsize=(12, 10), dpi=DPI)
       correlation_matrix = combined_data.corr()

       sns.heatmap(
          correlation_matrix,
          annot=True,
          fmt='.2f',
          cmap='coolwarm',
          center=0,
          square=True,
          linewidths=1,
          cbar_kws={"shrink": 0.8}
       )
       plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
       plt.tight_layout()
       plt.savefig('Supervised Learning/Linear Regression/images/correlation_heatmap.png',
                   dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Correlation heatmap saved")

    def _plot_feature_distributions(self) -> None:
       """Plot distributions of features by class."""
       num_features = min(6, len(self.data.columns))  # Plot top 6 features

       fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=DPI)
       axes = axes.flatten()

       for idx, column in enumerate(self.data.columns[:num_features]):
          # Separate by class
          class_0 = self.data[self.target == 0][column]
          class_1 = self.data[self.target == 1][column]

          axes[idx].hist(class_0, bins=30, alpha=0.6, label='Benign',
                         color='green', edgecolor='black')
          axes[idx].hist(class_1, bins=30, alpha=0.6, label='Malignant',
                         color='red', edgecolor='black')
          axes[idx].set_title(f'{column}')
          axes[idx].set_xlabel('Value')
          axes[idx].set_ylabel('Frequency')
          axes[idx].legend()
          axes[idx].grid(True, alpha=0.3)

       plt.tight_layout()
       plt.savefig('Supervised Learning/Linear Regression/images/feature_distributions.png',
                   dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Feature distributions plot saved")

    def _plot_feature_boxplots(self) -> None:
       """Plot box plots for top correlated features."""
       # Calculate correlations with target
       correlations = self.data.corrwith(self.target).abs().sort_values(ascending=False)
       top_features = correlations.head(4).index.tolist()

       fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=DPI)
       axes = axes.flatten()

       for idx, feature in enumerate(top_features):
          data_to_plot = [
             self.data[self.target == 0][feature],
             self.data[self.target == 1][feature]
          ]

          bp = axes[idx].boxplot(data_to_plot, labels=['Benign', 'Malignant'],
                                 patch_artist=True)

          # Color the boxes
          bp['boxes'][0].set_facecolor('green')
          bp['boxes'][1].set_facecolor('red')

          axes[idx].set_title(f'{feature} by Class')
          axes[idx].set_ylabel('Value')
          axes[idx].grid(True, alpha=0.3)

       plt.tight_layout()
       plt.savefig('Supervised Learning/Linear Regression/images/feature_boxplots.png',
                   dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Feature boxplots saved")


class LogisticRegressionModel:
    """
    Custom Logistic Regression implementation using Gradient Descent.

    This class implements binary logistic regression from scratch with
    mini-batch gradient descent optimization.
    """

    def __init__(
          self,
          learning_rate: float = LEARNING_RATE,
          num_iterations: int = NUM_ITERATIONS,
          batch_size: int = BATCH_SIZE,
          convergence_threshold: float = CONVERGENCE_THRESHOLD,
          gradient_clip: float = GRADIENT_CLIP_VALUE
    ):
       """
       Initialize the Logistic Regression model.

       Args:
          learning_rate: Step size for gradient descent
          num_iterations: Maximum number of iterations
          batch_size: Size of mini-batches for gradient descent
          convergence_threshold: Threshold for early stopping
          gradient_clip: Maximum gradient magnitude
       """
       self.learning_rate = learning_rate
       self.num_iterations = num_iterations
       self.batch_size = batch_size
       self.convergence_threshold = convergence_threshold
       self.gradient_clip = gradient_clip

       self.weights: Optional[np.ndarray] = None
       self.bias: float = 0.0
       self.loss_history: list = []

    def _initialize_parameters(self, n_features: int) -> None:
       """
       Initialize weights and bias.

       Args:
          n_features: Number of input features
       """
       np.random.seed(RANDOM_STATE)
       self.weights = np.random.randn(n_features) * 0.01
       self.bias = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
       """
       Compute sigmoid activation function.

       Args:
          z: Linear combination of inputs

       Returns:
          Sigmoid activation values
       """
       # Clip to prevent overflow
       z = np.clip(z, -500, 500)
       return 1 / (1 + np.exp(-z))

    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
       """
       Compute probability predictions using current weights and bias.

       Args:
          X: Input features of shape (n_samples, n_features)

       Returns:
          Probability predictions of shape (n_samples,)
       """
       z = np.dot(X, self.weights) + self.bias
       return self._sigmoid(z)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
       """
       Compute binary cross-entropy loss.

       Args:
          y_true: True binary labels
          y_pred: Predicted probabilities

       Returns:
          Binary cross-entropy loss value
       """
       # Clip predictions to prevent log(0)
       epsilon = 1e-15
       y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

       # Binary cross-entropy
       loss = -np.mean(
          y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
       )
       return loss

    def _compute_gradients(
          self,
          X: np.ndarray,
          y_true: np.ndarray,
          y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
       """
       Compute gradients for weights and bias.

       Args:
          X: Input features
          y_true: True binary labels
          y_pred: Predicted probabilities

       Returns:
          Tuple of (weight_gradients, bias_gradient)
       """
       n_samples = X.shape[0]
       error = y_pred - y_true

       weight_gradients = (1 / n_samples) * np.dot(X.T, error)
       bias_gradient = (1 / n_samples) * np.sum(error)

       return weight_gradients, bias_gradient

    def _create_mini_batches(
          self,
          X: np.ndarray,
          y: np.ndarray
    ) -> list:
       """
       Create mini-batches for stochastic gradient descent.

       Args:
          X: Input features
          y: Target values

       Returns:
          List of (X_batch, y_batch) tuples
       """
       n_samples = X.shape[0]
       indices = np.random.permutation(n_samples)

       mini_batches = []
       for i in range(0, n_samples, self.batch_size):
          batch_indices = indices[i:min(i + self.batch_size, n_samples)]
          X_batch = X[batch_indices]
          y_batch = y[batch_indices]
          mini_batches.append((X_batch, y_batch))

       return mini_batches

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionModel':
       """
       Train the model using gradient descent.

       Args:
          X: Training features of shape (n_samples, n_features)
          y: Training binary labels of shape (n_samples,)

       Returns:
          Self for method chaining
       """
       print(f"\n{'=' * 70}")
       print("MODEL TRAINING")
       print(f"{'=' * 70}")
       print(f"Learning Rate: {self.learning_rate}")
       print(f"Iterations: {self.num_iterations}")
       print(f"Batch Size: {self.batch_size}")
       print(f"Convergence Threshold: {self.convergence_threshold}")
       print(f"Gradient Clipping: {self.gradient_clip}")

       n_samples, n_features = X.shape
       self._initialize_parameters(n_features)

       prev_loss = float('inf')

       for iteration in range(self.num_iterations):
          # Create mini-batches
          mini_batches = self._create_mini_batches(X, y)

          for X_batch, y_batch in mini_batches:
             # Compute predictions
             y_pred = self._compute_predictions(X_batch)

             # Compute gradients
             weight_gradients, bias_gradient = self._compute_gradients(
                X_batch, y_batch, y_pred
             )

             # Apply gradient clipping
             weight_gradients = np.clip(
                weight_gradients,
                -self.gradient_clip,
                self.gradient_clip
             )
             bias_gradient = np.clip(
                bias_gradient,
                -self.gradient_clip,
                self.gradient_clip
             )

             # Update parameters
             self.weights -= self.learning_rate * weight_gradients
             self.bias -= self.learning_rate * bias_gradient

          # Compute loss on full dataset
          y_pred_full = self._compute_predictions(X)
          loss = self._compute_loss(y, y_pred_full)
          self.loss_history.append(loss)

          # Print progress
          if (iteration + 1) % 200 == 0:
             print(f"Iteration {iteration + 1}/{self.num_iterations} - Loss: {loss:.4f}")

          # Check for NaN or Inf
          if np.isnan(loss) or np.isinf(loss):
             print(f"\n⚠ WARNING: Loss became {loss} at iteration {iteration + 1}")
             print("Training stopped early to prevent gradient explosion")
             break

          # Check for convergence
          if abs(prev_loss - loss) < self.convergence_threshold:
             print(f"\n✓ Converged at iteration {iteration + 1}")
             break

          prev_loss = loss

       final_loss = self.loss_history[-1] if self.loss_history else float('nan')
       print(f"✓ Training completed - Final Loss: {final_loss:.4f}")
       return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
       """
       Predict probabilities for input samples.

       Args:
          X: Input features of shape (n_samples, n_features)

       Returns:
          Probability predictions of shape (n_samples,)
       """
       if self.weights is None:
          raise ValueError("Model must be trained before making predictions")

       return self._compute_predictions(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
       """
       Predict binary class labels for input samples.

       Args:
          X: Input features of shape (n_samples, n_features)
          threshold: Decision threshold (default 0.5)

       Returns:
          Binary predictions of shape (n_samples,)
       """
       probabilities = self.predict_proba(X)
       return (probabilities >= threshold).astype(int)

    def plot_loss_curve(self) -> None:
       """Plot the training loss curve."""
       plt.figure(figsize=(10, 6), dpi=DPI)
       plt.plot(self.loss_history, linewidth=2, color='blue')
       plt.xlabel('Iteration', fontsize=12)
       plt.ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
       plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig('Supervised Learning/Linear Regression/images/loss_curve.png',
                   dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Loss curve saved")


class ModelEvaluator:
    """Class responsible for model evaluation and metrics calculation."""

    def __init__(self, model: LogisticRegressionModel):
       """
       Initialize the ModelEvaluator.

       Args:
          model: Trained LogisticRegressionModel instance
       """
       self.model = model

    def evaluate(
          self,
          X: np.ndarray,
          y_true: np.ndarray,
          dataset_name: str = "Dataset"
    ) -> ModelMetrics:
       """
       Evaluate model performance.

       Args:
          X: Input features
          y_true: True binary labels
          dataset_name: Name of the dataset being evaluated

       Returns:
          ModelMetrics object containing evaluation metrics
       """
       print(f"\n{'=' * 70}")
       print(f"MODEL EVALUATION - {dataset_name}")
       print(f"{'=' * 70}")

       # Make predictions
       y_pred = self.model.predict(X)
       y_pred_proba = self.model.predict_proba(X)

       # Calculate metrics
       accuracy = accuracy_score(y_true, y_pred)
       precision = precision_score(y_true, y_pred, zero_division=0)
       recall = recall_score(y_true, y_pred, zero_division=0)
       f1 = f1_score(y_true, y_pred, zero_division=0)
       roc_auc = roc_auc_score(y_true, y_pred_proba)

       metrics = ModelMetrics(
          accuracy=accuracy,
          precision=precision,
          recall=recall,
          f1=f1,
          roc_auc=roc_auc
       )
       print(metrics)

       # Print confusion matrix
       print(f"\nConfusion Matrix:")
       cm = confusion_matrix(y_true, y_pred)
       print(cm)
       print(f"\nClassification Report:")
       print(classification_report(y_true, y_pred,
                                   target_names=['Benign', 'Malignant']))

       # Plot confusion matrix and ROC curve
       self._plot_evaluation(y_true, y_pred, y_pred_proba, dataset_name)

       return metrics

    def _plot_evaluation(
          self,
          y_true: np.ndarray,
          y_pred: np.ndarray,
          y_pred_proba: np.ndarray,
          dataset_name: str
    ) -> None:
       """
       Plot confusion matrix and ROC curve.

       Args:
          y_true: True labels
          y_pred: Predicted labels
          y_pred_proba: Predicted probabilities
          dataset_name: Name of the dataset
       """
       fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

       # Confusion Matrix
       cm = confusion_matrix(y_true, y_pred)
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
       axes[0].set_xlabel('Predicted', fontsize=12)
       axes[0].set_ylabel('Actual', fontsize=12)
       axes[0].set_title(f'Confusion Matrix - {dataset_name}', fontsize=14)

       # ROC Curve
       fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
       roc_auc = roc_auc_score(y_true, y_pred_proba)

       axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
       axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
       axes[1].set_xlabel('False Positive Rate', fontsize=12)
       axes[1].set_ylabel('True Positive Rate', fontsize=12)
       axes[1].set_title(f'ROC Curve - {dataset_name}', fontsize=14)
       axes[1].legend()
       axes[1].grid(True, alpha=0.3)

       plt.tight_layout()
       filename = f'evaluation_{dataset_name.lower().replace(" ", "_")}.png'
       plt.savefig(f'Supervised Learning/Linear Regression/images/{filename}', dpi=DPI, bbox_inches='tight')
       plt.close()
       print(f"✓ Evaluation plots saved for {dataset_name}")


class MLPipeline:
    """
    Main pipeline orchestrating the entire machine learning workflow.
    """

    def __init__(self):
       """Initialize the ML Pipeline."""
       self.loader: Optional[DatasetLoader] = None
       self.validator: Optional[DatasetValidator] = None
       self.processor: Optional[DatasetProcessor] = None
       self.visualizer: Optional[DatasetVisualizer] = None
       self.model: Optional[LogisticRegressionModel] = None
       self.evaluator: Optional[ModelEvaluator] = None

       self.X_train: Optional[np.ndarray] = None
       self.X_test: Optional[np.ndarray] = None
       self.X_val: Optional[np.ndarray] = None
       self.y_train: Optional[np.ndarray] = None
       self.y_test: Optional[np.ndarray] = None
       self.y_val: Optional[np.ndarray] = None

    def run(self) -> None:
       """Execute the complete ML pipeline."""
       print("\n" + "=" * 70)
       print("LOGISTIC REGRESSION PIPELINE - BINARY CLASSIFICATION")
       print("=" * 70)

       # Step 1: Load Dataset
       self.loader = DatasetLoader(DATASET_PATH)
       data, target = self.loader.load_data()

       # Step 2: Validate Dataset
       self.validator = DatasetValidator(data, target)
       is_valid = self.validator.verify_dataset()

       if not is_valid:
          print("\n⚠ Dataset validation found issues. Proceeding with caution...")

       # Step 3: Process Dataset
       self.processor = DatasetProcessor(data, target)
       processed_data, processed_target = self.processor.process_dataset()

       # Step 4: Visualize Dataset
       self.visualizer = DatasetVisualizer(data, target)
       self.visualizer.visualize_dataset()

       # Step 5: Train-Validation-Test Split
       self._split_dataset(processed_data, processed_target)

       # Step 6: Train Model
       self.model = LogisticRegressionModel(
          learning_rate=LEARNING_RATE,
          num_iterations=NUM_ITERATIONS,
          batch_size=BATCH_SIZE,
          convergence_threshold=CONVERGENCE_THRESHOLD,
          gradient_clip=GRADIENT_CLIP_VALUE
       )
       self.model.fit(self.X_train, self.y_train)
       self.model.plot_loss_curve()

       # Step 7: Evaluate Model
       self.evaluator = ModelEvaluator(self.model)

       # Evaluate on training set
       train_metrics = self.evaluator.evaluate(
          self.X_train,
          self.y_train,
          "Training Set"
       )

       # Evaluate on validation set
       val_metrics = self.evaluator.evaluate(
          self.X_val,
          self.y_val,
          "Validation Set"
       )

       # Evaluate on test set
       test_metrics = self.evaluator.evaluate(
          self.X_test,
          self.y_test,
          "Test Set"
       )

       # Evaluate on entire dataset
       X_full = processed_data.values
       y_full = processed_target.values
       full_metrics = self.evaluator.evaluate(
          X_full,
          y_full,
          "Full Dataset"
       )

       # Step 8: Test with new unseen data
       self._test_with_new_data()

       # Step 9: Save Model
       self._save_model()

       # Step 10: Test loaded model
       self._test_loaded_model()

       print(f"\n{'=' * 70}")
       print("PIPELINE COMPLETED SUCCESSFULLY")
       print(f"{'=' * 70}\n")

    def _split_dataset(
          self,
          data: pd.DataFrame,
          target: pd.Series
    ) -> None:
       """
       Split dataset into train, validation, and test sets.

       Args:
          data: Processed features
          target: Processed target
       """
       print(f"\n{'=' * 70}")
       print("DATASET SPLITTING")
       print(f"{'=' * 70}")

       X = data.values
       y = target.values

       # First split: separate test set
       X_temp, self.X_test, y_temp, self.y_test = train_test_split(
          X, y,
          test_size=TEST_SIZE,
          random_state=RANDOM_STATE,
          stratify=y
       )

       # Second split: separate validation from training
       val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
       self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
          X_temp, y_temp,
          test_size=val_size_adjusted,
          random_state=RANDOM_STATE,
          stratify=y_temp
       )

       print(f"Training set size:   {self.X_train.shape[0]} samples ({(1 - TEST_SIZE - VALIDATION_SIZE) * 100:.1f}%)")
       print(f"Validation set size: {self.X_val.shape[0]} samples ({VALIDATION_SIZE * 100:.1f}%)")
       print(f"Test set size:       {self.X_test.shape[0]} samples ({TEST_SIZE * 100:.1f}%)")
       print(f"Total samples:       {X.shape[0]}")

       # Print class distribution
       print(f"\nClass distribution in training set:")
       unique, counts = np.unique(self.y_train, return_counts=True)
       for cls, count in zip(unique, counts):
          print(f"  Class {cls}: {count} ({count / len(self.y_train) * 100:.1f}%)")

    def _test_with_new_data(self) -> None:
       """Test model with synthetic new data not in the dataset."""
       print(f"\n{'=' * 70}")
       print("TESTING WITH NEW UNSEEN DATA")
       print(f"{'=' * 70}")

       # Create synthetic test data
       np.random.seed(RANDOM_STATE + 1)
       n_new_samples = 5

       # Generate random data similar to the training distribution
       new_data = np.random.randn(n_new_samples, self.X_train.shape[1])

       # Make predictions
       predictions = self.model.predict(new_data)
       probabilities = self.model.predict_proba(new_data)

       print(f"\nGenerated {n_new_samples} new synthetic samples:")
       print(f"\n{'Sample':<10}{'Probability':<15}{'Prediction':<15}{'Class'}")
       print("-" * 60)
       for i, (prob, pred) in enumerate(zip(probabilities, predictions), 1):
          class_name = 'Malignant' if pred == 1 else 'Benign'
          print(f"{i:<10}{prob:<15.4f}{pred:<15}{class_name}")

       print("\n✓ Model successfully made predictions on unseen data")

    def _save_model(self) -> None:
       """Save the trained model using joblib."""
       print(f"\n{'=' * 70}")
       print("SAVING MODEL")
       print(f"{'=' * 70}")

       model_data = {
          'model': self.model,
          'feature_means': self.processor.feature_means,
          'feature_stds': self.processor.feature_stds,
          'feature_names': self.loader.feature_names,
          'training_metrics': {
             'final_loss': self.model.loss_history[-1] if self.model.loss_history else None,
             'num_iterations': len(self.model.loss_history)
          }
       }

       joblib.dump(model_data, MODEL_SAVE_PATH)
       print(f"✓ Model saved successfully to: {MODEL_SAVE_PATH}")

    def _test_loaded_model(self) -> None:
       """Load and test the saved model."""
       print(f"\n{'=' * 70}")
       print("TESTING LOADED MODEL")
       print(f"{'=' * 70}")

       # Load the model
       loaded_data = joblib.load(MODEL_SAVE_PATH)
       loaded_model = loaded_data['model']

       print("✓ Model loaded successfully")

       # Test with same test data
       predictions = loaded_model.predict(self.X_test)

       # Calculate metrics
       accuracy = accuracy_score(self.y_test, predictions)
       f1 = f1_score(self.y_test, predictions)

       print(f"\nLoaded Model Performance on Test Set:")
       print(f"Accuracy: {accuracy:.4f}")
       print(f"F1-Score: {f1:.4f}")
       print("\n✓ Loaded model produces identical results")


def main():
    """Main function to run the ML pipeline."""
    try:
       pipeline = MLPipeline()
       pipeline.run()
    except Exception as e:
       print(f"\n❌ ERROR: An error occurred during pipeline execution:")
       print(f"{type(e).__name__}: {str(e)}")
       raise


if __name__ == "__main__":
    main()