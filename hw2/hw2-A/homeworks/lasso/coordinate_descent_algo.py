from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    return 2 * np.sum(X ** 2, axis=0)


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    n, d = X.shape
    residuals = y - (X @ weight)
    b = np.mean(residuals)
    for k in range(d):
        c_k = 2 * np.sum(X[:, k] * (y - (b + X @ weight + (-weight[k] * X[:, k]))))
        if c_k < -_lambda:
            weight[k] = (c_k + _lambda) / a[k]
        elif c_k > _lambda:
            weight[k] = (c_k - _lambda) / a[k]
        else:
            # between -lambda to lamda
            weight[k] = 0

    return weight, b


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    predictions = X @ weight + bias
    mse = np.sum((y - predictions) ** 2)
    l1_penalty = _lambda * np.sum(np.abs(weight))
    return mse + l1_penalty


@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None

    weight = np.copy(start_weight)
    while old_w is None or not convergence_criterion(weight, old_w, convergence_delta):
        old_w = np.copy(weight)
        weight, bias = step(X, y, weight, a, _lambda)

    return weight, bias


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    return np.max(np.abs(weight - old_w)) < convergence_delta

def generate_synthetic_data(n=500, d=1000, k=100, sigma=1):
    np.random.seed(42)
    X = np.random.randn(n, d)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    true_w = np.zeros(d)
    for j in range(k):
        true_w[j] = (j + 1) / k
    
    epsilon = np.random.randn(n) * sigma
    y = X @ true_w + epsilon

    return X, y, true_w

def compute_lambda_max(X, y):
    n, d = X.shape
    y_mean = np.mean(y)
    return np.max(2 * np.abs(X.T @ (y - y_mean)))

def plot_lasso_path(lambdas, nonzeros):
    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, nonzeros, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Number of Nonzero Weights")
    plt.title("Lasso Regularization Path")
    plt.grid(True)
    plt.show(block=False)

def plot_fdr_tpr(fdr_values, tpr_values):
    plt.figure(figsize=(8, 6))
    plt.plot(fdr_values, tpr_values, marker='o', linestyle='-')
    plt.xlabel("False Discovery Rate (FDR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("FDR vs. TPR for Lasso Regularization Path")
    plt.grid(True)
    plt.show()


def lasso_regularization_metrics(X, y, true_w, lambda_max, num_lambdas=10, factor=2):
    lambdas = [lambda_max / (factor ** i) for i in range(num_lambdas)]
    nonzeros = []
    fdr_values = []
    tpr_values = []

    weight = np.zeros(X.shape[1])

    for _lambda in lambdas:
        weight, _ = train(X, y, _lambda, start_weight=weight)

        nonzero_indices = np.where(weight != 0)[0]  # Features selected by Lasso
        true_nonzero_indices = np.where(true_w != 0)[0]  # True relevant features

        num_false_discoveries = np.sum(np.isin(nonzero_indices, true_nonzero_indices, invert=True))
        num_true_positives = np.sum(np.isin(nonzero_indices, true_nonzero_indices))

        total_nonzeros = len(nonzero_indices)
        fdr = num_false_discoveries / total_nonzeros if total_nonzeros > 0 else 0
        tpr = num_true_positives / len(true_nonzero_indices) if len(true_nonzero_indices) > 0 else 0

        nonzeros.append(total_nonzeros)
        fdr_values.append(fdr)
        tpr_values.append(tpr)

    return lambdas, nonzeros, fdr_values, tpr_values


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    X, y, true_w = generate_synthetic_data(n=500, d=1000, k=100, sigma=1)
    lambda_max = compute_lambda_max(X, y)

    lambdas, nonzeros, fdr_values, tpr_values = lasso_regularization_metrics(X, y, true_w, lambda_max)

    # Part a
    plot_lasso_path(lambdas, nonzeros)

    # Part b
    plot_fdr_tpr(fdr_values, tpr_values)


if __name__ == "__main__":
    main()
