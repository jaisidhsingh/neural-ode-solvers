import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import *
from tqdm import tqdm
import matplotlib.pyplot as plt


class NeuralFunctionPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.act(self.input_layer(x))
        return self.output_layer(x)


def actual_trajectory(x, initial_condition):
    """
    Returns the trajector after solving the ODE analytically
    y = y_0 + integral_{0}^x (x**2 + torch.sin(x)**2 dx)
    """
    return (1/3)*x**3 + 0.5 * x - 0.25 * torch.sin(2*x) + initial_condition


def ode_function(x):
    """
    Returns the slope of the trajectory function,
    dy(x)/dx = f(x, y(x))
    """
    return x**2 + torch.sin(x)**2


def plot_eval(x, y, y_pred):
    plt.plot(x, y, label="actual")
    plt.plot(x, y_pred, label="predicted")
    plt.legend()
    plt.show()


def train_node():
    """
    Here x is a vector and so y is also a vector.
    """
    x_dim = 1
    hidden_dim = 5
    y_dim = 1
    model = NeuralFunctionPredictor(x_dim, hidden_dim, y_dim)

    x = torch.linspace(0, 100, steps=int(1e+4))
    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ic = 0.0
    y = actual_trajectory(x, ic)

    model.train()
    bar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        x = x.T.unsqueeze(-1)
        optimizer.zero_grad()
        y_pred = model(x)

        dy_pred = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True)

        slope_residual = (ode_function(x) - dy_pred)**2
        initial_condition_residual = (ic - y_pred[0])**2
        loss = slope_residual + initial_condition_residual

        running_loss = loss.item()

        loss.backward()
        optimizer.step()
        
        bar.set_postfix({"epoch": epoch+1, "loss": running_loss})
        bar.update(1)

    model.eval()
    y_pred = model(x)
    plot_eval(x, y, y_pred)


if __name__ == "__main__":
    train_node()
