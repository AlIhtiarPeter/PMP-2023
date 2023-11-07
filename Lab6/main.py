import pymc as pm
import arviz as az
from matplotlib import pyplot as plt

def func():
    Y_values = [0, 5, 10]
    t_values = [0.2, 0.5]
    model = pm.Model()
    traces = {}

    for Y in Y_values:
        for t in t_values:
            with model:
                n = pm.Poisson(('n' + str(Y) + str(t)), mu=10)
                obs = pm.Binomial(('obs' + str(Y) + str(t)), n=n, p=t, observed=Y)
                trace = pm.sample(100)
            traces[(Y, t)] = trace

    fig, axes = plt.subplots(len(Y_values), len(t_values), figsize=(12, 8))
    for i in range(len(Y_values)):
        for j in range(len(t_values)):
            az.plot_posterior(traces[(Y_values[i], t_values[j])], var_names=[('n' + str(Y_values[i]) + str(t_values[j]))], ax=axes[i, j])
            axes[i, j].set_title(f'Y = {Y_values[i]}, t = {t_values[j]}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    func()
