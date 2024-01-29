import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

Titanic = pd.read_csv('/content/Titanic.csv')
Titanic = pd.DataFrame(Titanic)
Ages = Titanic["Age"].values
Survived = Titanic["Survived"].values
Pclass = Titanic["Pclass"].values
proc_ages = []
proc_survived = []
proc_pclass = []
for i in range(len(Ages)):
    if Ages[i] >= 0 and Pclass[i] != None and Survived[i] != None:
        proc_ages.append(Ages[i])
        proc_survived.append(Survived[i])
        proc_pclass.append(Pclass[i])

proc_ages = np.array(proc_ages)
proc_survived = np.array(proc_survived)
proc_pclass = np.array(proc_pclass)

ages_mean = proc_ages.mean()
ages_std = proc_ages.std()
proc_ages = (proc_ages - ages_mean) / ages_std
X = np.column_stack((proc_ages, proc_pclass))

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
    X_shared = pm.MutableData('x_shared', X)
    mu = pm.Deterministic('μ', alpha + pm.math.dot(X_shared, beta))
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
    bd = pm.Deterministic("bd", -alpha / beta[1] - beta[0] / beta[1] * proc_ages)
    y_pred = pm.Bernoulli("y_pred", p=theta, observed=proc_survived)
    idata = pm.sample(500, return_inferencedata=True)


plt.scatter(proc_ages, proc_pclass, c=[f"C{x}" for x in proc_survived])
plt.xlabel("ages")
plt.ylabel("pclass")

plt.show()

preds = []
preds1 = []
for i in range(len(proc_ages)):
  obs_std1 = [proc_ages[i], proc_pclass[i]]
  sigmoid = lambda x: 1 / (1 + np.exp(-x))
  posterior_g = idata.posterior.stack(samples={"chain", "draw"})
  mu = posterior_g['alpha'] + posterior_g['beta'][0]*obs_std1[0] + posterior_g['beta'][1]*obs_std1[1]
  theta = sigmoid(mu)
  pred = theta.values.mean()
  preds1.append(pred)
  if(pred < 0.5):
    preds.append(0)
  else:
    preds.append(1)


plt.scatter(proc_ages, proc_pclass, c=[f"C{x}" for x in preds])
plt.xlabel("ages")
plt.ylabel("pclass")

plt.show()


plt.plot(preds1)
plt.show()

#1c
#Din grafic rezulta ca pclass influenteaza mai mult rezultatul.
#Daca clasa este mai mare, probabilitatea ca survived sa fie egal cu 1 scade
#Daca pclass este mai mare ca unu sansele de supravietuire sunt mereu mai mici de 0.5

obs_std1 = [(30 - ages_mean) / ages_std, 2]
sigmoid = lambda x: 1 / (1 + np.exp(-x))
posterior_g = idata.posterior.stack(samples={"chain", "draw"})
mu = posterior_g['alpha'] + posterior_g['beta'][0] * obs_std1[0] + posterior_g['beta'][1] * obs_std1[1]
theta = sigmoid(mu)
az.plot_posterior(theta.values, hdi_prob=0.9)

