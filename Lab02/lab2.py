import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

#1
x = stats.expon.rvs(scale = 1/4, size=10000)
y = stats.expon.rvs(scale = 1/6, size=10000)
z = 0.4 * x + 0.6 * y
az.plot_posterior({'x':x,'y':y,'z':z})
plt.show()

#2
a = stats.gamma.rvs(4,0,1/3,size=10000)
b = stats.gamma.rvs(4,0,1/2,size=10000)
c = stats.gamma.rvs(5,0,1/2,size=10000)
d = stats.gamma.rvs(5,0,1/3,size=10000)
lat = stats.expon.rvs(scale = 1/4, size=10000)
gamma =  0.25 * a + 0.25 * b + 0.3 * c + 0.2 * d + lat
az.plot_posterior({'a':a,'b':b,'c':c,'d':d,'lat' : lat,'gamma' : gamma})
plt.show()

#3
ar_ss = []
ar_sb = []
ar_bs = []
ar_bb = []


for i in range(100):
    ss = 0
    sb = 0
    bs = 0
    bb = 0
    z1 = np.random.uniform(0,1,10);
    z2 = np.random.uniform(0,1,10);
    for j in range(10):
        if z1[j] < 0.5:
            if z2[j] < 0.3:
                ss = ss + 1
            else:
                sb = sb + 1
        else:
            if z2[j] < 0.3:
                bs = bs + 1
            else:
                bb = bb +1
    ar_ss.append(ss)
    ar_sb.append(sb)
    ar_bs.append(bs)
    ar_bb.append(bb)
az.plot_posterior({'ss':ar_ss,'sb':ar_sb,'bs':ar_bs,'bb':ar_bb})
plt.show()