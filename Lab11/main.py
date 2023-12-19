import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt


def func():
    clusters = 3
    n_cluster = [166, 167, 167]
    n_total = sum(n_cluster)
    means = [5, 2.5, 0]
    std_devs = [2, 1.5, 1]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    plt.show()

    with pm.Model() as model2:
      weights = pm.Dirichlet('weights', np.ones(2))
      means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), 2), sigma=10, shape=2)
      std_devs = pm.Uniform('std_devs', lower=0, upper=10, shape=2)
      data_likelihood = pm.NormalMixture('data_likelihood', w=weights, mu=means, sigma=std_devs, observed=mix)
      trace2 = pm.sample(1, return_inferencedata=True)

    with pm.Model() as model3:
      weights = pm.Dirichlet('weights', np.ones(3))
      means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), 3), sigma=10, shape=3)
      std_devs = pm.Uniform('std_devs', lower=0, upper=10, shape=3)
      data_likelihood = pm.NormalMixture('data_likelihood', w=weights, mu=means, sigma=std_devs, observed=mix)
      trace3 = pm.sample(1, return_inferencedata=True)

    with pm.Model() as model4:
      weights = pm.Dirichlet('weights', np.ones(4))
      means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), 4), sigma=10, shape=4)
      std_devs = pm.Uniform('std_devs', lower=0, upper=10, shape=4)
      data_likelihood = pm.NormalMixture('data_likelihood', w=weights, mu=means, sigma=std_devs, observed=mix)
      trace4 = pm.sample(1, return_inferencedata=True)

    pm.compute_log_likelihood(trace2, model=model2)
    pm.compute_log_likelihood(trace3, model=model3)
    pm.compute_log_likelihood(trace4, model=model4)
    cmp_df = az.compare({'model2':trace2, 'model3':trace3, 'model4':trace4},
                       method='BB-pseudo-BMA', ic="waic", scale="deviance")
    az.plot_compare(cmp_df)
    cmp_df2 = az.compare({'model2':trace2, 'model3':trace3, 'model4':trace4},
                         method='BB-pseudo-BMA', ic="loo", scale="deviance")
    az.plot_compare(cmp_df2)


if __name__ == '__main__':
    func()