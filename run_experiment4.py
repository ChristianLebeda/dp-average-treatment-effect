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
            print("IPW logistic regression:")
            print(ipwlog)
            print("G-formula linear regression:")
            print(glin)
            print("AIPW logistic and linear regression:")
            print(aipwloglin)
            print()
            print()

        features, data, tau = data_generator.experiment4()

        # Useful for testing. Compute IPW with oracle access to propensity score.
        #ipworacle.append(float(estimator.oracle_IPW(data)))

        # IPW estimator
        if k <= 200: # IPW quickly becomes unstable for this experiment. We don't need large k for the plot
            ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'IPW', binary = binary)
            ensemble.fit_estimators(data, estimator.LogisticRegressionModel(), None)
            ipwlog.append(ensemble.evaluate(estimate_variance = False))
        
        #G-formula estimators
        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'G-formula', binary = binary)
        ensemble.fit_estimators(data, None, estimator.LinearRegressionModel())
        glin.append(ensemble.evaluate(estimate_variance = False))

        # AIPW estimator
        if k <= 200: # For the plot we show that AIPW is stable in the regime where IPW became unstable
            ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'AIPW', binary = binary)
            ensemble.fit_estimators(data, estimator.LogisticRegressionModel(), estimator.LinearRegressionModel())
            aipwloglin.append(ensemble.evaluate(estimate_variance = False))

    
    print(f"Finishing run with settings: n {n}. k {k}. b {b}. binary data {binary}. eta {eta}. sigma {sigma}. repetitions {rep}")


n = 20000
b = 1
eta = 0.1
sigma = 0 # We don't add noise when running the experiments. Noise is added when plotting
binary = True

for k in [2, 25, 50, 75, 100, 200, 300, 400, 500]: 
    ipworacle = []
    ipwlog = []
    glin = []
    aipwloglin = []
    
    experiment(100, n, k, b, binary, eta, sigma)

    print("Results")

    #print("Oracle IPW")
    #print(ipworacle)
    #print()

    print("IPW logistic regression:")
    print(ipwlog)
    print()

    print("G-formula linear regression:")
    print(glin)
    print()

    print("AIPW logistic and linear regression:")
    print(aipwloglin)
    print()

    