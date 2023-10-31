import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def importData(path):
    data = pd.read_csv(path)
    set1 = [data.values[i][1] for i in range(0, 180)]
    set2 = [data.values[i][1] for i in range(180, 240)]
    set3 = [data.values[i][1] for i in range(240, 720)]
    set4 = [data.values[i][1] for i in range(720, 900)]
    set5 = [data.values[i][1] for i in range(900, 1200)]
    return (set1, set2, set3, set4, set5)


model = pm.Model()
data = importData("trafic.csv")

with model:
    interval1 = data[0]
    lambda1 = np.mean(interval1)
    min1 = np.min(interval1)
    max1 = np.max(interval1)

    interval2 = data[1]
    lambda2 = np.mean(interval2)
    min2 = np.min(interval2)
    max2 = np.max(interval2)

    interval3 = data[2]
    lambda3 = np.mean(interval3)
    min3 = np.min(interval3)
    max3 = np.max(interval3)

    interval4 = data[3]
    lambda4 = np.mean(interval4)
    min4 = np.min(interval4)
    max4 = np.max(interval4)

    interval5 = data[4]
    lambda5 = np.mean(interval5)
    min5 = np.min(interval5)
    max5 = np.max(interval5)

    idx = np.arange(1200)
    lambda_ = pm.math.switch(idx < 180 , lambda1, pm.math.switch(idx < 240, lambda2,pm.math.switch(idx < 720,lambda3,pm.math.switch(idx < 900,lambda4,lambda5))))
    set = interval1+interval2+interval3+interval4+interval5
    observation = pm.Poisson("obs", lambda_, observed=set)


