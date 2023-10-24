import scipy.stats as sc
import matplotlib.pyplot as plt
import arviz as az

clientCount = sc.poisson.rvs(20, size=1000)
orderTime = sc.norm.rvs(loc=2, scale=0.5, size=1000)
cookTime = sc.expon.rvs(scale=4.33, size=1000)

az.plot_posterior({'ClientCount':clientCount,'orderTime':orderTime,'cookTime':cookTime})
plt.show()

def over15avg1(alpha):
    count = 0
    orderTime = sc.norm.rvs(loc=2, scale=0.5, size=10000)
    cookTime = sc.expon.rvs(scale=alpha, size=10000)
    serviceTime = orderTime + cookTime
    for i in serviceTime:
        if(i < 15):
            count += 1
    return count/10000

def over15avg2(alpha):
    p = sc.expon.cdf(x=13, scale=alpha)
    return p

alpha = 1/60
while(over15avg1(alpha) > 0.95):
    alpha += 1/60
alpha -= 1/60

print(alpha)

def avgTime(alpha):
    orderTime = sc.norm.rvs(loc=2, scale=0.5, size=1000)
    cookTime = sc.expon.rvs(scale=alpha, size=1000)
    serviceTime = orderTime + cookTime
    return sum(serviceTime) / 1000

print(avgTime(alpha))