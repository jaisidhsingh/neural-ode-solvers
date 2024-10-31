import jax
import jax.numpy as jnp


class MLP:
    def __init__(self, layer_sizes, key):
        self.params = []
        keys = jax.random.split(key, len(layer_sizes) - 1)
        
        for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
            w = jax.random.normal(k, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
            b = jnp.zeros(out_dim)
            self.params.append((w, b))
    
    def __call__(self, x):
        for w, b in self.params[:-1]:
            x = jnp.dot(x, w) + b
            x = jax.nn.tanh(x)
        w, b = self.params[-1]
        return jnp.dot(x, w) + b


def actual_trajectory(x, initial_condition):
    """
    Returns the analytical solution of the ODE
    y = y_0 + 0_/^x x**2 + jnp.sin(x)**2
    y = y_0 + 0.5*x + 0.25*jnp.cos(2*x)
    """
    return initial_condition + (x**3) / 3 + 0.5*x - 0.25*jnp.sin(2*x)


def ode_function(x):
    """
    Returns the slope of the trajectory as defined by the ODE
    dy(x)/dx = f(x, y(x))
    """
    return x**2 + jnp.sin(x)**2


def loss_function(params, x, model, ic):
    slope_pred = jax.grad(model)(x)
    slope_loss = (slope_pred - ode_function(x))**2
    ic_loss = (model(x[0]) - ic)**2
    loss = (slope_loss + ic_loss)**2
    return loss


def update_params(params, grads, lr):
    new_params = {}
    for k, v in params.items():
        new_params[k] = v - lr*grads[k]
    return new_params


def train_node():
    num_data_points = 1000
    x_dim = 10
    y_dim = 5
    x = jnp.random.normal(num_data_points, x_dim)

    layer_sizes = [x_dim, 128, 64, y_dim]
    key = jax.random.PRNGKey(0)
    model = MLP(layer_sizes, key)

    num_epochs = 20
    initial_condition = jnp.random.normal(10,)
    lr = 1e-3

    loss_grad = jax.jit(jax.grad(loss_function, argnums=(0,)))
    for epoch in range(num_epochs):
        loss = loss_function(model.params, x, model, initial_condition)
        grads = loss_grad(model.params, x, model, initial_condition)
        update_params(model.params, grads, lr)


