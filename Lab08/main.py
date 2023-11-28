import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def importData(path):
    data = pd.read_csv(path)
    return [(data.loc[:,"Price"][i],data.loc[:,"Speed"][i],data.loc[:,"HardDrive"][i],data.loc[:,"Ram"][i],data.loc[:,"Premium"][i]) for i in range(len(data.loc[:,"Price"]))]

def func():
    price, speed, hardDrive, ram, premium = zip(*importData("/Prices.csv"))
    price = np.array(price)
    speed = np.array(speed)
    hardDrive = np.array(hardDrive)
    ram = np.array(ram)
    premium = np.array(premium)
    with pm.Model() as model:
        y = pm.Data('price', price)
        x1 = pm.Data('mhz', speed)
        x2 = pm.Data('hard_drive', hardDrive)
        a = pm.Normal('α', mu=0, sigma=1)
        b1 = pm.Normal('β1', mu=0, sigma=1)
        b2 = pm.Normal('β2', mu=0, sigma=1)
        s = pm.Normal('ε', 1)
        c = a + x1 * b1 + pm.math.log(x2) * b2
        y_pred = pm.Normal('price_predicted', mu=c, sigma=s, observed=y)
        d_price = pm.Deterministic('d', a + 33 * b1 + pm.math.log(540) * b2)
        idata = pm.sample(2000)

    az.plot_trace(idata, var_names=['β1', 'β2','d'])

    b_p1 = idata.posterior['β1'].mean().item()
    b_p2 = idata.posterior['β2'].mean().item()
    a_p = idata.posterior['α'].mean().item()
    delta = np.abs(price - (speed * b_p1 + np.log(hardDrive) * b_p2 + a_p))
    delta = np.array(delta)
    print(np.mean(delta))
if __name__ == '__main__':
    func()

