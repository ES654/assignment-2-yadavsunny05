import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time
from metrics import *
import seaborn as sns

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(X, y)
    y_hat = LR.predict(X)
    # LR.plot()

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y)) 



LR.plot_residuals()


n_vals = [i for i in range(30,2000)]
p_vals = [i for i in range(10,1000)]

emperical_n = []
P = 5
for N in n_vals:
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    startTime = time.time()
    LR = LinearRegression(fit_intercept=False)
    LR.fit(X,y)
    endTime = time.time()
    emperical_n.append((endTime-startTime))

emerical_p = list()
N = 1000
for P in p_vals:
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    startTime = time.time()
    LR = LinearRegression(fit_intercept=False)
    LR.fit(X,y)
    endTime = time.time()
    emerical_p.append((endTime-startTime))

fig = plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.scatter(N_range, fit_timesVsN)
plt.xlabel("N(Number of Instances)")
plt.ylabel("Emperical fit time")
plt.subplot(1,2,2)
plt.scatter(P_range, fit_timesVsP)
plt.xlabel("P (Features)")
plt.ylabel("Emperical fit time")
plt.show()


