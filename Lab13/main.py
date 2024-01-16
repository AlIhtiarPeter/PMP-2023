import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

centered_eight_data = az.load_arviz_data("centered_eight")
non_centered_eight_data = az.load_arviz_data("non_centered_eight")

def ex1():
    print("Modelul Centrat:")
    print("Numărul de lanțuri:", centered_eight_data.posterior.chain.shape[0])
    print("Mărimea totală a eșantionului generat:", centered_eight_data.posterior.draw.shape[0])

    print("\nModelul Necentrat:")
    print("Numărul de lanțuri:", non_centered_eight_data.posterior.chain.shape[0])
    print("Mărimea totală a eșantionului generat:", non_centered_eight_data.posterior.draw.shape[0])
    print("\n")

    az.plot_posterior(centered_eight_data)
    az.plot_posterior(non_centered_eight_data)
    plt.show()


def ex2():
    summaries = pd.concat([az.summary(centered_eight_data, var_names=["mu", "tau"]),
                           az.summary(non_centered_eight_data, var_names=["mu", "tau"])])
    print(summaries["r_hat"])
    print("\n")

    az.plot_autocorr(centered_eight_data, var_names=["mu", "tau"])
    az.plot_autocorr(non_centered_eight_data, var_names=["mu", "tau"])
    plt.show()


def ex3():
    print(centered_eight_data.sample_stats.diverging.sum())
    print(non_centered_eight_data.sample_stats.diverging.sum())

    az.plot_pair(centered_eight_data, var_names=["mu", "tau"], divergences=True)
    az.plot_pair(non_centered_eight_data, var_names=["mu", "tau"], divergences=True)

    plt.show()


ex1()
ex2()
ex3()
