if __name__ == "__main__":
    from coordinate_descent_algo import train, compute_lambda_max  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

def plot_nonzero_weights(lambdas, nonzeros):
    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, nonzeros, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Number of Nonzero Weights")
    plt.title("Lasso Regularization Path")
    plt.grid(True)
    plt.show(block=False)


def plot_regularization_paths(lambdas, coef_paths):
    plt.figure(figsize=(8, 6))
    for feature, path in coef_paths.items():
        plt.plot(lambdas, path, marker='o', linestyle='-', label=feature)

    plt.xscale('log')
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Coefficient Value")
    plt.title("Regularization Paths for Selected Features")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def compute_mse(X, y, weight, bias):
    predictions = X @ weight + bias
    return np.mean((predictions - y) ** 2)

def plot_mse_vs_lambda(lambdas, mse_train, mse_test):
    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, mse_train, marker='o', linestyle='-', label="Train MSE")
    plt.plot(lambdas, mse_test, marker='o', linestyle='-', label="Test MSE")

    plt.xscale('log')
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE vs. Lambda on Training and Test Data")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_analyze_lasso(X_train, y_train, X_test, y_test, df_train):
    lambda_max = compute_lambda_max(X_train, y_train)
    num_lambdas = 10
    lambdas = [lambda_max / (2**i) for i in range(num_lambdas)]

    nonzeros = []
    mse_train = []
    mse_test = []
    selected_features = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    coef_paths = {feature: [] for feature in selected_features}

    weight = np.zeros(X_train.shape[1])
    for _lambda in lambdas:
        weight, bias = train(X_train, y_train, _lambda, start_weight=weight)

        nonzeros.append(np.count_nonzero(weight))
        mse_train.append(compute_mse(X_train, y_train, weight, bias))
        mse_test.append(compute_mse(X_test, y_test, weight, bias))

        for feature in selected_features:
            coef_paths[feature].append(weight[df_train.columns[1:].tolist().index(feature)])

    plot_nonzero_weights(lambdas, nonzeros)
    plot_regularization_paths(lambdas, coef_paths)
    plot_mse_vs_lambda(lambdas, mse_train, mse_test)

def analyze_largest_coefficients(X_train, y_train, df_train, lambda_value=30):
    weight, _ = train(X_train, y_train, _lambda=lambda_value)
    
    feature_names = df_train.columns[1:].tolist()
    max_feature = feature_names[np.argmax(weight)]
    min_feature = feature_names[np.argmin(weight)]
    
    print(f"Largest positive coefficient feature: {max_feature}")
    print(f"Largest negative coefficient feature: {min_feature}")


@problem.tag("hw2-A", start_line=3)
def main():
    df_train, df_test = load_dataset("crime")

    X_train, y_train = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    X_test, y_test = df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values

    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std  

    train_and_analyze_lasso(X_train, y_train, X_test, y_test, df_train)
    analyze_largest_coefficients(X_train, y_train, df_train, lambda_value=30)


if __name__ == "__main__":
    main()
