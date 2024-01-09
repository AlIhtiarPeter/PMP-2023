import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = abs(grid - 0.5)  # uniform prior
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def ex1():
    data = np.repeat([0, 1], (10, 3))
    points = 10
    h = data.sum()
    t = len(data) - h
    grid, posterior = posterior_grid(points, h, t)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ');


def computeError(count):
    errors = []
    for _ in range(1000):
        N = count
        x, y = np.random.uniform(-1, 1, size=(2, N))
        inside = (x ** 2 + y ** 2) <= 1
        pi = inside.sum() * 4 / N
        error = abs((pi - np.pi) / pi) * 100
        errors.append(error)

    mean = sum(errors) / 1000
    a = [(e - mean) * (e - mean) for e in errors]
    deviation = sum(a) / 1000
    print(mean)
    print(deviation)
    return mean, deviation


def ex2():
    m1, d1 = computeError(10)
    m2, d2 = computeError(100)
    m3, d3 = computeError(1000)
    m4, d4 = computeError(10000)
    m5, d5 = computeError(100000)
    means = [m1, m2, m3, m4, m5]
    deviations = [d1, d2, d3, d4, d5]
    plt.plot(means)
    plt.show()
    plt.plot(deviations)
    plt.show()


def metropolis(func, draws=10000):
    """A very simple Metropolis implementation"""
    trace = np.zeros(draws)
    old_x = func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


def ex3():
    beta_params = [(1, 1), (20, 20), (1, 4)]
    for a,b in beta_params:
        func = stats.beta(a,b)
        trace = metropolis(func=func)
        x = np.linspace(0.01, .99, 100)
        y = func.pdf(x)
        plt.xlim(0, 1)
        plt.plot(x, y, 'C1-', lw=3, label='True distribution')
        plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
        plt.xlabel('x')
        plt.ylabel('pdf(x)')
        plt.yticks([])
        plt.legend()
        plt.show()