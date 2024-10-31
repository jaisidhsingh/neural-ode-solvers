import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import *


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
    return 0.5 * x - 0.25 * torch.cos(2*x) + initial_condition


def ode_function(x):
    """
    Returns the slope of the trajectory function,
    dy(x)/dx = f(x, y(x))
    """
    y = torch.sin(x)
    return x**2 + y**2


def train_node():
    """
    Here x is a vector and so y is also a vector.
    """
    x_dim = 10
    hidden_dim = 5
    y_dim = 10
    model = NeuralFunctionPredictor(x_dim, hidden_dim, y_dim)

    dataset = torch.randn(100, x_dim)
    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        x = dataset
        optimizer.zero_grad()
        y_pred = model(x)

        dy_pred = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True)
        ic_pred = model(x[0])

        slope_residual = (ode_function(x) - y_pred)**2
        initial_condition_residual = (actual_trajectory(x[0]) - ic_pred)**2
        loss = slope_residual + initial_condition_residual

        loss.backward()
        optimizer.step()



