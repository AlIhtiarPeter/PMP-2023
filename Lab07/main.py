import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import numpy as np

def func():
    data = pd.read_csv("/auto-mpg.csv")
    mpg = data.loc[:,"mpg"]
    hp = data.loc[:,"horsepower"]
    plt.scatter(hp, mpg)
    plt.xlabel('hp')
    plt.ylabel('mpg', rotation=0)
    plt.show()

    with pm.Model() as model:
      a = pm.Normal('α', mu=mpg.mean(), sigma=1)
      b = pm.Normal('β', mu=0, sigma=1)
      mu = a + b * hp
      e = pm.HalfCauchy('ε', 5)
      mpg_predicted = pm.Normal('mpg_predicted', mu=mu, sigma=e, observed=mpg)
      idata = pm.sample(2000, return_inferencedata=True)

    plt.plot(hp, mpg, 'C0.')
    posterior_g = idata.posterior.stack(samples={"chain", "draw"})
    alpha_m = posterior_g['α'].mean().item()
    beta_m = posterior_g['β'].mean().item()
    draws = range(0, posterior_g.samples.size, 1)
    plt.plot(hp, posterior_g['α'][draws].values
              + posterior_g['β'][draws].values * hp[:,None],
              c='gray', alpha=0.5)
    plt.plot(hp, alpha_m + beta_m * hp, c='k',
              label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    func()

#In concluzie daca horsepower va creste miles per gallon va scadea. Cand puterea masinii va fi mai mare, consumul va fi mai mare care va rezulta in mai putine mile per galon