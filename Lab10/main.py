import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def func(order,sd,path):
    dummy_data = np.loadtxt(path)
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    with pm.Model() as model_p:
        x_1s = pm.Data("x1", x_1s)
        y_1s = pm.Data("y1", y_1s)
        alpha = pm.Normal('α', mu=0, sigma=1)
        beta = pm.Normal('β', mu=0, sigma=sd, shape=order)
        e = pm.HalfNormal('ε', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s)
        idata_p = pm.sample(750, return_inferencedata=True)

    alpha_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()


def func3():
    dummy_data = np.loadtxt("/content/dummy.csv")
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    x_1p = np.vstack([x_1 ** i for i in range(1, 1 + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    with pm.Model() as model_l:
        alpha = pm.Normal('α', mu=0, sigma=1)
        beta = pm.Normal('β', mu=0, sigma=10, shape=1)
        e = pm.HalfNormal('ε', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s)
        idata_l = pm.sample(750, return_inferencedata=True)

    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    x_1p = np.vstack([x_1 ** i for i in range(1, 2 + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    with pm.Model() as model_p:
        alpha = pm.Normal('α', mu=0, sigma=1)
        beta = pm.Normal('β', mu=0, sigma=10, shape=2)
        e = pm.HalfNormal('ε', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s)
        idata_p = pm.sample(750, return_inferencedata=True)

    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    x_1p = np.vstack([x_1 ** i for i in range(1, 3 + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    with pm.Model() as model_c:
        alpha = pm.Normal('α', mu=0, sigma=1)
        beta = pm.Normal('β', mu=0, sigma=10, shape=3)
        e = pm.HalfNormal('ε', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s)
        idata_c = pm.sample(750, return_inferencedata=True)

    pm.compute_log_likelihood(idata_l,model=model_l)
    pm.compute_log_likelihood(idata_p,model=model_p)
    pm.compute_log_likelihood(idata_c,model=model_c)
    cmp_df = az.compare({'model_l':idata_l,'model_p':idata_p, 'model_c':idata_c},
                       method='BB-pseudo-BMA', ic="waic", scale="deviance")
    az.plot_compare(cmp_df)
    cmp_df2 = az.compare({'model_l': idata_l, 'model_p': idata_p, 'model_c': idata_c},
                         method='BB-pseudo-BMA', ic="loo", scale="deviance")
    az.plot_compare(cmp_df2)

if __name__ == '__main__':
    func3()