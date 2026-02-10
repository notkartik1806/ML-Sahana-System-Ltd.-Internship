import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class DataLoader:

    def load_data(self, path):
        data = pd.read_csv(path)
        print("Dataset Loaded Successfully")
        return data


class DataVerification:

    def verify_data(self, data):
        print("\nDataset Info:\n")
        print(data.info())

        print("\nMissing Values:\n")
        print(data.isnull().sum())


class DataProcessing:

    def preprocess_data(self, data):
        data = data.drop("Unnamed: 0", axis=1)
        print("\nUnwanted Column Removed")
        return data


class DataVisualization:

    def plot_graph(self, x, y):

        plt.scatter(x, y)
        plt.xlabel("Years Of Experience")
        plt.ylabel("Salary")
        plt.title("Salary Prediction Dataset")
        plt.show()


class LinearRegressionModel:

    def train_model(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        print("\nModel Training Completed")

        return model, X_test, y_test


class ModelValidation:

    def validate_model(self, model, X_test, y_test):

        predictions = model.predict(X_test)

        accuracy = r2_score(y_test, predictions)

        print("\nModel Accuracy:", accuracy)


if __name__ == "__main__":

    loader = DataLoader()
    data = loader.load_data("Salary_dataset.csv")

    verify = DataVerification()
    verify.verify_data(data)

    process = DataProcessing()
    clean_data = process.preprocess_data(data)

    X = clean_data[['YearsExperience']]
    y = clean_data['Salary']

    visual = DataVisualization()
    visual.plot_graph(X, y)

    model_train = LinearRegressionModel()
    model, X_test, y_test = model_train.train_model(X, y)

    check = ModelValidation()
    check.validate_model(model, X_test, y_test)