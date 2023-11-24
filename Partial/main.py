import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def ex1_1():
    first_player = np.random.rand(20000)
    p0_count = 0
    for i in first_player:
        if i < 0.5: # incepe p0
            p0 = np.random.rand(1)
            n = 0
            if p0 > 1/3:
                n = 1
            p1 = np.random.rand(n+1)
            m = 0
            for throw in p1:
                if throw > 0.5:
                    m += 1
            if n >= m:
                p0_count += 1
        else: #incepe p1
            p1 = np.random.rand(1)
            n = 0
            if p1 > 1 / 2:
                n = 1
            p0 = np.random.rand(n + 1)
            m = 0
            for throw in p0:
                if throw > 1/3:
                    m += 1
            if n < m:
                p0_count += 1
    p1_count = 20000 - p0_count
    if p0_count > p1_count:
        print("Jucatorul P0 are sanse mai mari sa castige. Sanse :" + str(p0_count/20000))
    elif p1_count > p0_count:
        print("Jucatorul P1 are sanse mai mari sa castige. Sanse :" + str(p1_count/20000))
    else:
        print("Jucatorii au sanse egale de castig")


def ex1_2():
    model = BayesianNetwork([('P0', 'W'), ('P1', 'W')])
    CPD_P0 = TabularCPD(variable="P0", variable_card=2, values=[[0.333], [0.667]])
    CPD_P1 = TabularCPD(variable="P1", variable_card=2, values=[[0.5], [0.5]])
    CPD_first = TabularCPD(variable="first", variable_card=2, values=[[0.5], [0.5]])
    CPD_R1 = TabularCPD(variable='R1', variable_card=2,
                       values=[[0.333, 0.5],
                               [0.667, 0.5]],
                       evidence=['first'],
                       evidence_card=[2])
    CPD_R2 = TabularCPD(variable='R2', variable_card=3,
                        values=[[0.5, 0.25, 0.333, 0.112],
                                [0.5, 0.5, 0.667, 0.444],
                                [0, 0.25, 0, 0.444]],
                        evidence=['first', 'R1'],
                        evidence_card=[2,  2])


def ex2(sigma,mean):
    data = np.random.normal(mean, sigma, 200)
    az.plot_posterior({'x': data})
    plt.show()
    with pm.Model() as model:
        mean_time = pm.Poisson('poisson', mu=10)
        trace = pm.sample(200)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ex2(0.1,10)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
