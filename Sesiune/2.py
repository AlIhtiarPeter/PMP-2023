import numpy as np
import scipy.stats as stats

def predict():
  N = 10000
  x = stats.geom.rvs(0.3, size=N)
  y = stats.geom.rvs(0.5, size=N)
  inside = x > y**2
  return inside.sum()/N

k = 30
predictions = []
for _ in range(k):
  predictions.append(predict())

predictions = np.array(predictions)
mean = predictions.mean()
std = predictions.std()
print(mean)
print(std)