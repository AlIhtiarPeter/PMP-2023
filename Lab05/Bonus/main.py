import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az

def cookTime():
    cooktime1 = np.random.exponential(3, 100)
    meanTime = np.mean(cooktime1)

    with pm.Model() as model:
        alpha = 3
        cookTime2 = pm.Exponential("cookTime", 1/alpha)
        trace = pm.sample(100, tune=100)

    summary = az.summary(trace)
    print(summary)

    print(meanTime)

if __name__ == '__main__':
    cookTime()
