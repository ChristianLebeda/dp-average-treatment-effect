from estimator import EnsembleEstimator
import estimator
import data_generator
from time import gmtime, strftime
import numpy as np
import pandas as pd

np.random.seed(42)

def experiment(rep, n, k, b, binary, eta, sigma, print_progress = True):
    print("Settings for experiment")
    print(f"n {n}. k {k}. b {b}. binary data {binary}. eta {eta}. sigma {sigma}. repetitions {rep}")
    print ("Starttime ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    for i in range(rep):
        features, data, tau = data_generator.experiment4(n)
        
        #G-formula estimators
        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'G-formula', binary = binary)
        ensemble.fit_estimators(data, None, estimator_SA.LinearRegressionModel())
        glin.append(ensemble.evaluate(estimate_variance = False))
        # ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'G-formula', binary = binary)
        # ensemble.fit_estimators(data, None, estimator.LinearRegressionModel())
        # glin.append(ensemble.evaluate(estimate_variance = False))

    
    #print(f"Finishing run with settings: n {n}. k {k}. b {b}. binary data {binary}. eta {eta}. sigma {sigma}. repetitions {rep}")


n = 20000
b = 1
eta = 0.1
sigma = 0 # We don't add noise when running the experiments. Noise is added when plotting
binary = True

for k in [2, 50, 100, 150, 200, 300, 400]: 
    glin = []
    
    experiment(100, n, k, b, binary, eta, sigma)

    print("Results")

    print("G-formula linear regression:")
    print(glin)
    print()
    