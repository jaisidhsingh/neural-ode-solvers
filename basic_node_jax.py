import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layer_sizes, key):
        self.params = []
        keys = jax.random.split(key, len(layer_sizes) - 1)
        
        for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
            w = jax.random.normal(k, (in_dim, out_dim)) * jnp.sqrt(2.0 / in_dim)
            b = jnp.zeros(out_dim)
            self.params.append((w, b))
        
        self.mode = "train"

    def train(self):
        self.mode = "train"
    
    def eval(self):
        self.mode = "eval"

    def __call__(self, x):
        for w, b in self.params[:-1]:
            x = jnp.dot(x, w) + b
            x = jax.nn.gelu(x)
        w, b = self.params[-1]
        out = jnp.dot(x, w) + b
        return out[0] if self.mode == "train" else out


def actual_trajectory(x, initial_condition):
    """
    Returns the analytical solution of the ODE
    """
    return initial_condition + (1/3) * (x**3)


def ode_function(x):
    """
    Returns the slope of the trajectory as defined by the ODE
    dy(x)/dx = f(x, y(x))
    """
    return x**2


def loss_function(params, x, model, ic, x_0):
    slope_pred = jax.grad(model)(x)
    slope_loss =  (slope_pred - ode_function(x))**2
    ic_loss =  (model(x_0) - ic)**2
    loss = (slope_loss + ic_loss) / 2
    return loss[0]


def update_params(params, grads, lr):
    new_params = []
    for (w, b), (grad_w, grad_b) in zip(params, grads[0]):
        w_new = w - lr*grad_w
        b_new = b - lr*grad_b
        new_params.append((w_new, b_new))
    return tuple(new_params) 


def plot_eval(x, y, y_pred):
    plt.plot(x, y, label="actual")
    plt.plot(x, y_pred, label="predicted")
    plt.legend()
    plt.show()


def train_node():
    upper_bound = 1
    x_dim = 1
    y_dim = 1
    x = jnp.linspace(0, upper_bound, 1000)

    layer_sizes = [x_dim, 128, 64, y_dim]
    key = jax.random.PRNGKey(0)
    model = MLP(layer_sizes, key)
    model.train()

    num_epochs = 1
    initial_condition = 0.0 
    lr = 1e-3

    loss_grad = jax.grad(loss_function, argnums=(0,))
    
    x = x.reshape(x.shape[0], 1)
    x_0 = x[0]
    bar = tqdm(total=num_epochs * 1000)
    for epoch in range(num_epochs):
        for idx, point in enumerate(x):
            loss_value = loss_function(model.params, point, model, initial_condition, x_0)

            grads = loss_grad(model.params, point, model, initial_condition, x_0)
            new_params = update_params(model.params, grads, lr)
            model.params = new_params

            bar.set_postfix({"Step": epoch * 50 + idx, "loss": loss_value.item()})
            bar.update(1)

    bar.close()

    model.eval()
    y_pred = model(x).reshape(1000,)
    x = x.reshape(1000,)
    y = actual_trajectory(x, initial_condition)
    plot_eval(x, y, y_pred)


if __name__ == "__main__":
    train_node()

