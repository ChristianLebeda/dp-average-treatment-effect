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
            print("IPW decision tree:")
            print(ipwtree)
            print("G-formula linear regression:")
            print(glin)
            print("G-formula decision tree:")
            print(gtree)
            print("AIPW logistic and linear regression:")
            print(aipwloglin)
            print("AIPW logistic regression and decision tree:")
            print(aipwlogtree)
            print("AIPW decision tree and linear regression:")
            print(aipwtreelin)
            print("AIPW decision trees:")
            print(aipwtrees)
            print()
            print()

        features, data, tau = data_generator.experiment2()

        # Useful for testing. Compute IPW with oracle access to propensity score.
        #ipworacle.append(float(estimator.oracle_IPW(data)))

        # IPW estimators
        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'IPW', binary = binary)
        ensemble.fit_estimators(data, estimator.LogisticRegressionModel(), None)
        ipwlog.append(ensemble.evaluate(estimate_variance = False))

        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'IPW', binary = binary)
        ensemble.fit_estimators(data, estimator.DecisionTreeModel(), None)
        ipwtree.append(ensemble.evaluate(estimate_variance = False))
        
        #G-formula estimators
        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'G-formula', binary = binary)
        ensemble.fit_estimators(data, None, estimator.LinearRegressionModel())
        glin.append(ensemble.evaluate(estimate_variance = False))
        
        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'G-formula', binary = binary)
        ensemble.fit_estimators(data, None, estimator.DecisionTreeModel())
        gtree.append(ensemble.evaluate(estimate_variance = False))

        # AIPW estimators
        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'AIPW', binary = binary)
        ensemble.fit_estimators(data, estimator.LogisticRegressionModel(), estimator.LinearRegressionModel())
        aipwloglin.append(ensemble.evaluate(estimate_variance = False))

        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'AIPW', binary = binary)
        ensemble.fit_estimators(data, estimator.LogisticRegressionModel(), estimator.DecisionTreeModel())
        aipwlogtree.append(ensemble.evaluate(estimate_variance = False))
              
        ensembleAIPW2 = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'AIPW', binary = binary)
        ensembleAIPW2.fit_estimators(data, estimator.DecisionTreeModel(), estimator.LinearRegressionModel())
        aipwtreelin.append(ensemble.evaluate(estimate_variance = False))
    
        ensemble = EnsembleEstimator(n, k, features, b = b, eta = eta, sigma = sigma, estimator = 'AIPW', binary = binary)
        ensemble.fit_estimators(data, estimator.DecisionTreeModel(), estimator.DecisionTreeModel())
        aipwtrees.append(ensemble.evaluate(estimate_variance = False))

    
    print(f"Finishing run with settings: n {n}. k {k}. b {b}. binary data {binary}. eta {eta}. sigma {sigma}. repetitions {rep}")


n = 250000
b = 1
eta = 0.2
sigma = 0 # We don't add noise when running the experiments. Noise is added when plotting
binary = False

for k in [500]: # Note: Run with a smaller value for K if you want a fast approximation of the experiment
    ipworacle = []
    ipwlog = []
    ipwtree = []
    glin = []
    gtree = []
    aipwloglin = []
    aipwlogtree = []
    aipwtreelin = []
    aipwtrees = []
    
    experiment(100, n, k, b, binary, eta, sigma)

    print("Results")

    #print("Oracle IPW")
    #print(ipworacle)
    #print()

    print("IPW logistic regression:")
    print(ipwlog)
    print()

    print("IPW decision tree:")
    print(ipwtree)
    print()

    print("G-formula linear regression:")
    print(glin)
    print()

    print("G-formula decision tree:")
    print(gtree)
    print()

    print("AIPW logistic and linear regression:")
    print(aipwloglin)
    print()

    print("AIPW logistic regression and decision tree:")
    print(aipwlogtree)
    print()

    print("AIPW decision tree and linear regression:")
    print(aipwtreelin)
    print()

    print("AIPW decision trees:")
    print(aipwtrees)
    print()

    