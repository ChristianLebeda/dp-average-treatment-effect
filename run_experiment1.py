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
        if print_progress and i % 5 == 0:
            print('Started run number', i)
            print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            
            #print("ipworacle =",ipworacle)
            print("glin =",glin)
            print("aipwloglin =",aipwloglin)
            print()

        features, data, tau = data_generator.experiment1()

        # Useful for testing. Compute IPW with oracle access to propensity score.
        #ipworacle.append(float(estimator.oracle_IPW(data)))

        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'G-formula', binary = binary)
        ensemble.fit_estimators(data, None, estimator.LinearRegressionModel())
        glin.append(ensemble.evaluate(estimate_variance = False))

        ensembleAIPW3 = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'AIPW', binary = binary)
        ensembleAIPW3.fit_estimators(data, estimator.LogisticRegressionModel(), estimator.LinearRegressionModel())
        aipwloglin.append(ensembleAIPW3.evaluate(estimate_variance = False))

    
    print(f"Finishing run with settings: n {n}. k {k}. b {b}. binary data {binary}. eta {eta}. sigma {sigma}. repetitions {rep}")


n = 5000
b = 1
eta = 0.2
sigma = 0 # We don't add noise when running the experiments. Noise is added when plotting
binary = False

for k in [200]:
    ipworacle = []
    glin = []
    aipwloglin = []
    
    experiment(100, n, k, b, binary, eta, sigma)

    print("Results")

    #print("Oracle IPW")
    #print(ipworacle)
    #print()

    print("G-formula linear regression:")
    print(glin)
    print()

    print("AIPW logistic and linear regression:")
    print(aipwloglin)
    print()

    