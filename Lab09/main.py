import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(100)


def importData(path):
    data = pd.read_csv(path)
    return ([data.loc[:, "Admission"][i] for i in range(len(data.loc[:, "Admission"]))],
            [[data.loc[:, "GRE"][i], data.loc[:, "GPA"][i]] for i in range(len(data.loc[:, "GPA"]))])


admission, x = importData("/content/Admission.csv")
x = np.array(x)


def plot_gre_gpa(idata, gre, gpa):
    probabilities = []
    beta0 = idata.posterior['β0'][1]
    beta = idata.posterior['β'][1]
    for i in range(len(beta0)):
        probabilities.append(1 / (1 + np.exp(-(beta0[i] + gre * beta[i][0] + gpa * beta[i][1]))))
    probabilities = np.array(probabilities)

    hdi = pm.stats.hdi(probabilities, hdi_prob=0.9)
    print(hdi)
    hdi_probabilities = []
    for p in probabilities:
        if p >= hdi[0] and p <= hdi[1]:
            hdi_probabilities.append(p)
    hdi_probabilities = np.array(hdi_probabilities)
    hdi_probabilities = np.sort(hdi_probabilities)
    plt.figure()
    plt.plot(hdi_probabilities)
    plt.title(f'GRE = {gre} GPA = {gpa}')
    plt.show()


def func():
    with pm.Model() as model_1:
        beta0 = pm.Normal('β0', mu=0, sigma=10)
        beta = pm.Normal('β', mu=0, sigma=2, shape=2)
        mu = beta0 + pm.math.dot(x, beta)
        theta = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-mu)))
        bd = pm.Deterministic('bd', -beta0 / beta[0] - beta[1] / beta[0] * x[:, 1])
        yl = pm.Bernoulli('yl', p=theta, observed=admission)
        idata = pm.sample(2000, random_seed=rng, return_inferencedata=True)

    idx = np.argsort(x[:, 1])
    bd = idata.posterior['bd'].mean(("chain", "draw"))[idx]
    print(bd.mean())
    plt.scatter(x[:, 1], x[:, 0], c=[f'C{x}' for x in admission])
    plt.plot(x[:, 1][idx], bd, color='k');
    az.plot_hdi(x[:, 1], idata.posterior['bd'], color='k')
    plt.xlabel("gpa")
    plt.ylabel("gre")
    plt.show()

    plot_gre_gpa(idata, 550, 3.5)
    plot_gre_gpa(idata, 500, 3.2)
    plot_gre_gpa(idata, 1000, 4.5)


if __name__ == '__main__':
    func()
