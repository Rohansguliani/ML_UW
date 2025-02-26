# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()

        alpha_0 = 1 / math.sqrt(d)
        self.W0 = Parameter(Uniform(-alpha_0, alpha_0).sample((d, h)))
        self.b0 = Parameter(Uniform(-alpha_0, alpha_0).sample((h,)))

        alpha_1 = 1 / math.sqrt(h)
        self.W1 = Parameter(Uniform(-alpha_1, alpha_1).sample((h, k)))
        self.b1 = Parameter(Uniform(-alpha_1, alpha_1).sample((k,)))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        # First layer
        hidden = relu(x @ self.W0 + self.b0)
        # Second layer
        output = hidden @ self.W1 + self.b1
        return output


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()

        alpha_0 = 1 / math.sqrt(d)
        self.W0 = Parameter(Uniform(-alpha_0, alpha_0).sample((d, h0)))
        self.b0 = Parameter(Uniform(-alpha_0, alpha_0).sample((h0,)))
        alpha_1 = 1 / math.sqrt(h0)
        self.W1 = Parameter(Uniform(-alpha_1, alpha_1).sample((h0, h1)))
        self.b1 = Parameter(Uniform(-alpha_1, alpha_1).sample((h1,)))
        alpha_2 = 1 / math.sqrt(h1)
        self.W2 = Parameter(Uniform(-alpha_2, alpha_2).sample((h1, k)))
        self.b2 = Parameter(Uniform(-alpha_2, alpha_2).sample((k,)))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        hidden1 = relu(x @ self.W0 + self.b0)
        hidden2 = relu(hidden1 @ self.W1 + self.b1)
        output = hidden2 @ self.W2 + self.b2
        return output
    
def run_model(
    model: Module,
    optimizer: Adam,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
) -> Tuple[List[float], float, float, int]:
    losses = train(model, optimizer, train_loader)
    plt.plot(losses, label=f"{model_name} Training Loss")
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            loss = cross_entropy(outputs, y)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
    test_accuracy = correct / total
    test_loss = total_loss / len(test_loader)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    print(f"{model_name} Test Loss: {test_loss:.4f}")
    print(f"{model_name} Total Parameters: {total_params}")
    return losses, test_accuracy, test_loss, total_params


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    losses = []
    for epoch in range(100):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = cross_entropy(outputs, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        if accuracy >= 0.99:
            break
    return losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)

    d, h, k = 784, 64, 10
    model_f1 = F1(h=h, d=d, k=k)
    optimizer_f1 = Adam(model_f1.parameters(), lr=0.001)
    #run_model(model_f1, optimizer_f1, train_loader, test_loader, "F1")

    d, h0, h1, k = 784, 128, 64, 10
    model_f2 = F2(h0=h0, h1=h1, d=d, k=k)
    optimizer_f2 = Adam(model_f2.parameters(), lr=0.001)
    run_model(model_f2, optimizer_f2, train_loader, test_loader, "F2")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss for F2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
