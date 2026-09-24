from estimator import EnsembleEstimator
import estimator
import data_generator
from time import gmtime, strftime
import numpy as np
import pandas as pd

np.random.seed(42)

# Comparison non-private G-formula with private G-forumala with k=n/100
def experiment(rep, n, k, b, binary, eta, sigma):
    for i in range(rep):
        features, data, _ = data_generator.experiment3(n)

        #G-formula estimators
        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'G-formula', binary = binary)
        ensemble.fit_estimators(data, None, estimator.LogisticRegressionModel())
        glog.append(ensemble.evaluate(estimate_variance = False))
    
    print(f"Finishing run with settings: n {n}. k {k}. b {b}. binary data {binary}. eta {eta}. sigma {sigma}. repetitions {rep}")

b = 1
eta = 0.1
sigma = 0 # We don't add noise when running the experiments. Noise is added when plotting
binary = True

for n, k in [(10000, 2), (10000, 100), (15000, 150), (20000, 200)]:
    glog = []
    
    experiment(100, n, k, b, binary, eta, sigma)

    print("result:", glog)


    