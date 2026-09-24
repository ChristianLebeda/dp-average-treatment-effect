import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# This code is just used for the subsample and aggregate experiment
# Data is split in 2k groups, which are paired to give estimates for k folds with cross-fitting

# Aggregate using harmonic means for \hat{pi}_0 and \hat{pi}_1
def aggregatepropensities(data, idx, k, features, pi, eta, clip = True):
    propensities = [np.clip(pi[idx + k].propensity(data, features), eta, 1 - eta)] #[np.clip(pi[i].propensity(data, features), eta, 1 - eta) if clip else pi[i].propensity(data, features) for i in range(k) if i != idx] # Skip own index for cross-validation

    propensities_df = pd.DataFrame(np.column_stack(propensities), index=data.index)

    inverse_avg = (1 / propensities_df).mean(axis=1)
    inverse_avg_df = pd.DataFrame(inverse_avg, index=propensities_df.index, columns=["pihatinv"])
    tildepropensities_df = 1 - propensities_df

    tilde_inverse_avg = (1 / tildepropensities_df).mean(axis=1)
    tilde_inverse_avg_df = pd.DataFrame(tilde_inverse_avg, index=tildepropensities_df.index, columns=["pitildeinv"])
    
    # inverse_avg_df contains 1/\hat{pi}_1(i) from the paper
    # tilde_inverse_avg_df contains 1/(1 - \hat{pi}_0(i))
    
    return inverse_avg_df, tilde_inverse_avg_df

# Aggregrate average for fold idx
def aggregateregression(data, idx, k, features, mu, b, clip = True):
    mu0 = [np.clip(mu[idx + k].control(data, features), -b, b)]#[np.clip(mu[i].control(data, features), -b, b) if clip else mu[i].control(data, features) for i in range(k) if i != idx] # Skip own index for cross-validation
    mu1 = [np.clip(mu[idx + k].treated(data, features), -b, b)]#[np.clip(mu[i].treated(data, features), -b, b) if clip else mu[i].treated(data, features) for i in range(k) if i != idx] # Skip own index for cross-validation

    mu0_df = pd.DataFrame(np.column_stack(mu0), index=data.index)
    mu1_df = pd.DataFrame(np.column_stack(mu1), index=data.index)

    mu0_avg = (mu0_df).mean(axis=1)
    mu0_avg_df = pd.DataFrame(mu0_avg, index=mu0_avg.index, columns=["mu0hat"])
    mu1_avg = (mu1_df).mean(axis=1)
    mu1_avg_df = pd.DataFrame(mu1_avg, index=mu1_avg.index, columns=["mu1hat"])

    return mu0_avg_df, mu1_avg_df

# Estimators for the ATE and variance
def gformula(data, muhat0, muhat1):
    return muhat1["mu1hat"].sum() - muhat0["mu0hat"].sum()

def gformula_square(data, muhat0, muhat1, tauhat):
    return ((muhat1["mu1hat"] - muhat0["mu0hat"] - tauhat)**2).sum()

def IPW(data, pihatinv, pitildeinv):
    ipw_df = data['Y'] * data['A'] * pihatinv["pihatinv"] - data['Y'] * (1 - data['A']) * pitildeinv["pitildeinv"]
    return ipw_df.sum()

def IPW_square(data, pihatinv, pitildeinv, tauhat):
    ipw_df = data['Y'] * data['A'] * pihatinv["pihatinv"] - data['Y'] * (1 - data['A']) * pitildeinv["pitildeinv"]
    return ((ipw_df - tauhat)**2).sum()

def AIPW(data, pihatinv, pitildeinv, muhat0, muhat1):
    regression = muhat1["mu1hat"] - muhat0["mu0hat"]
    correction = (data['Y'] - muhat1["mu1hat"]) * data['A'] * pihatinv["pihatinv"] - (data['Y'] - muhat0["mu0hat"]) * (1 - data['A']) * pitildeinv["pitildeinv"]
    return regression.sum() + correction.sum()

def AIPW_square(data, pihatinv, pitildeinv, muhat0, muhat1, tauhat):
    regression = muhat1["mu1hat"] - muhat0["mu0hat"]
    correction = (data['Y'] - muhat1["mu1hat"]) * data['A'] * pihatinv["pihatinv"] - (data['Y'] - muhat0["mu0hat"]) * (1 - data['A']) * pitildeinv["pitildeinv"]
    return (((regression + correction) - tauhat)**2).sum()

# Oracle AIPW estimator that has access to the true propensity score and expected outcomes
def oracle_AIPW(data):
    if not data['synthetic'][0] == 1:
        print('ERROR: oracle AIPW estimator used for non-synthetic data')
        exit(-1)
    regression = data['mu1'] - data['mu0']
    correction = (data['Y'] - data['mu1']) * data['A'] / data['pi'] - (data['Y'] - data['mu0']) * (1 - data['A']) / (1 - data['pi'])
    AIPW_dataframe = regression + correction
    return AIPW_dataframe.mean()

# Oracle IPW estimator that has access to the true propensity score
def oracle_IPW(data):
    if not data['synthetic'][0] == 1:
        print('ERROR: oracle IPW estimator used for non-synthetic data')
        exit(-1)
    IPW_dataframe = data['Y'] * data['A'] / data['pi'] - data['Y'] * (1 - data['A']) / (1 - data['pi'])
    return IPW_dataframe.mean()

# Returns sample average treatment
def oracle_SATE(data):
    if not data['synthetic'][0] == 1:
        print('ERROR: oracle estimator used for non-synthetic data')
        exit(-1)
    ITE_dataframe = data['Emu1'] - data['Emu0']
    return ITE_dataframe.mean()


class EnsembleEstimator:
    # Initialize ensemble estimator
    # n = total number of data points
    # k = number of splits
    # features = names of all features
    # eta = overlap parameter, used for clipping
    # b = outcome space is [-b,b]
    # sigma = noise required for desired privacy with sensitivity 1. By setting sigma = 0 we can evaluate the non-private variant
    # estimator = should be one of ['G-formula', 'IPW', 'AIPW']
    # binary = indicates is outcomes are from {0,1}
    def __init__(self, n, k, features, eta = 0.1, b = 1, sigma = 0, estimator = 'AIPW', binary = False):
        self.n = n
        self.k = k
        self.features = features
        self.eta = eta
        self.b = b
        self.model_size = n // (2 * k) # Size of the small groups. Some have 1 more element
        self.sigma = sigma
        if estimator not in ['G-formula', 'IPW', 'AIPW']:
            print('ERROR: Unsupported estimator', estimator)
            exit(-1)
        self.estimator = estimator 
        self.binary = binary
        

    # This can be useful if we don't want to retrain the models.
    # Update - for the experiments we just set sigma = 0. We add noise when plotting data
    def update_noise_multiplier(self, sigma):
        self.sigma = sigma

    # Conservative sensitivity from Theorem 1. 
    # The sensitivity is slightly lower in some cases such as binary data.
    def sensitivity(self): 
        if self.estimator == 'G-formula':
            C = 16 * self.b * self.b
        elif self.estimator == 'IPW':
            C = 4 * self.b * self.b / (self.eta * self.eta)
        elif self.estimator == 'AIPW':
            C = 16 * self.b * self.b * (1 + 1 / self.eta)**2
        else:
            print('ERROR: sensitivity for estimator', self.estiamator ,'not supported')
            return 0

        return (C * (1 / self.n + 1 / (self.k - 1))**2)**0.5

    # This function takes as input the full dataset and functions for training the estimators. 
    def fit_estimators(self, data, propensity_trainer, regression_trainer):
        # Split the dataset into K folds
        self.groups = [data.loc[i] for i in np.array_split(data.index, self.k * 2)]

        # Sanity check that the parameters are set correctly
        for group in self.groups:
            if len(group) != self.model_size and len(group) != self.model_size + 1:
                print('ERROR: Unexpected group size. This should never happen, there might be a bug in the code')
                print(f'Excepted size around {self.n}//{self.k}={self.model_size}. Found {len(group)}')
                exit(-1)

        # Fit propensity score estimators
        if self.estimator != 'G-formula':
            self.pi = [propensity_trainer.fit_propensity(subset, self.features) for subset in self.groups]

        # Fit regression estimators
        if self.estimator != 'IPW':
            self.mu = [regression_trainer.fit_regression(subset, self.features, self.binary) for subset in self.groups]
    
    def evaluate(self, estimate_variance = True):
        if not hasattr(self, 'groups'):
            print("ERROR: Missing groups attribute during evaluate call. Make sure you called fit_estimators before evaluate")
            exit(-1)

        if self.estimator != 'G-formula' and not hasattr(self, 'pi'):
            print("ERROR: Missing propensity score estimators for evaluate call. Make sure you called fit_estimators before evaluate")
            exit(-1)

        if self.estimator != 'IPW' and not hasattr(self, 'mu'):
            print("ERROR: Missing regression estimators for evaluate call. Make sure you called fit_estimators before evaluate")
            exit(-1)
        
        tau = 0

        k = self.k

        for i in range(k): # Perform aggregation over each fold, ignoring models trained on that fold for cross-fitting purposes
            fold = self.groups[i]
            if self.estimator != 'G-formula':
                piinv, pitildeinv = aggregatepropensities(fold, i, k, self.features, self.pi, self.eta)

            if self.estimator != 'IPW':
                mu0, mu1 = aggregateregression(fold, i, k, self.features, self.mu, self.b)
            
            # Use the aggregate nuisance estimates in the ATE estimator
            if self.estimator == 'G-formula':
                tau += gformula(fold, mu0, mu1)
            if self.estimator == 'IPW':
                tau += IPW(fold, piinv, pitildeinv)
            if self.estimator == 'AIPW':
                tau += AIPW(fold, piinv, pitildeinv, mu0, mu1)

            fold = self.groups[i + k]
            if self.estimator != 'G-formula':
                piinv, pitildeinv = aggregatepropensities(fold, i, -k, self.features, self.pi, self.eta)

            if self.estimator != 'IPW':
                mu0, mu1 = aggregateregression(fold, i, -k, self.features, self.mu, self.b)
            
            # Use the aggregate nuisance estimates in the ATE estimator
            if self.estimator == 'G-formula':
                tau += gformula(fold, mu0, mu1)
            if self.estimator == 'IPW':
                tau += IPW(fold, piinv, pitildeinv)
            if self.estimator == 'AIPW':
                tau += AIPW(fold, piinv, pitildeinv, mu0, mu1)

        tau /= self.n # Normalize

        if estimate_variance:
            squarederror = 0

            for i in range(k):
                fold = self.groups[i]
                if self.estimator != 'G-formula':
                    piinv, pitildeinv = aggregatepropensities(fold, i, k, self.features, self.pi, self.eta)

                if self.estimator != 'IPW':
                    mu0, mu1 = aggregateregression(fold, i, k, self.features, self.mu, self.b)
                
                if self.estimator == 'G-formula':
                    squarederror += gformula_square(fold, mu0, mu1, tau)
                if self.estimator == 'IPW':
                    squarederror += IPW_square(fold, piinv, pitildeinv, tau)
                if self.estimator == 'AIPW':
                    squarederror += AIPW_square(fold, piinv, pitildeinv, mu0, mu1, tau)

        # print('The sensitivity is', self.sensitivity())
        
        if self.sigma: # add noise for privacy
            tau += np.random.normal(scale = self.sigma * self.sensitivity())

        return (float(tau), float(squarederror)) if estimate_variance else float(tau)



# Oracle estimators useful for debugging and baselines
class OraclePropensityScore:
    def propensity(self, data, features):
        return data['pi']

class OracleRegression:
    def control(self, data, features):
        return data['mu0']
    def treated(self, data, features):
        return data['mu1']

# Mostly for testing purposes. Uses the true propensity score and expected value for regression
# Can only be used for synthetic data
class OracleModel:
    def __init__(self):
        pass
    def fit_propensity(self, data, features):
        return OraclePropensityScore()
    def fit_regression(self, data, features):
        return OracleRegression()

# Wrapper classes for nuisance estimators
class PropensityScoreWrapper:
    def __init__(self, function):
        self.pi_hat = function
    def propensity(self, data, features):
        return self.pi_hat(data[features])

class RegressionWrapper:
    def __init__(self, function0, function1):
        self.mu0_hat = function0
        self.mu1_hat = function1
    def control(self, data, features):
        return self.mu0_hat(data[features])
    def treated(self, data, features):
        return self.mu1_hat(data[features])


# Used to fit logistic regression model for either propensity score or potential outcomes
class LogisticRegressionModel:
    def fit_propensity(self, data, features):
        prop_model = LogisticRegression()
        prop_model.fit(data[features], data['A'])
        return PropensityScoreWrapper(lambda x : prop_model.predict_proba(x)[:, 1])

    def fit_regression(self, data, features, binary = True):
        if not binary:
            print("Error: LogisticRegressionModel called with non-binary outcomes")
            exit(-1)
        treatment = data['A']
        y0_model = LogisticRegression()
        y1_model = LogisticRegression()

        y0_model.fit(data[treatment == 0][features], data[treatment == 0]['Y']) 
        y1_model.fit(data[treatment == 1][features], data[treatment == 1]['Y']) 

        mu_0 = lambda x : y0_model.predict_proba(x)[:, 1]
        mu_1 = lambda x : y1_model.predict_proba(x)[:, 1]  
        return RegressionWrapper(mu_0, mu_1)


# Standard linear regression for potential outcomes
class LinearRegressionModel:
    def fit_regression(self, data, features, binary = False):
        treatment = data['A']

        y0_model = LinearRegression().fit(data[treatment == 0][features], data[treatment == 0]['Y']) 
        y1_model = LinearRegression().fit(data[treatment == 1][features], data[treatment == 1]['Y'])  

        mu_0 = lambda x : y0_model.predict(x[features])  
        mu_1 = lambda x : y1_model.predict(x[features])  
        return RegressionWrapper(mu_0, mu_1)


# Used to fit random forest classifiers and regressors from sklearn
class RandomForestModel:
    def fit_propensity(self, data, features):
        prop_model = RandomForestClassifier(n_estimators=50, max_depth=3)
        prop_model.fit(data[features], data['A'])

        return PropensityScoreWrapper(lambda x : prop_model.predict_proba(x)[:, 1])

    def fit_regression(self, data, features, binary = True):
        treatment = data['A']
        if binary: # Use classifier for predicting probabilities for binary data, regressor with scalar data
            y0_model = RandomForestClassifier(n_estimators=50, max_depth=3).fit(data[treatment == 0][features], data[treatment == 0]['Y']) 
            y1_model = RandomForestClassifier(n_estimators=50, max_depth=3).fit(data[treatment == 1][features], data[treatment == 1]['Y'])  

            mu_0 = lambda x : y0_model.predict_proba(x[features])[:, 1] 
            mu_1 = lambda x : y1_model.predict_proba(x[features])[:, 1]
        else:
            y0_model = RandomForestRegressor(n_estimators=50, max_depth=3).fit(data[treatment == 0][features], data[treatment == 0]['Y']) 
            y1_model = RandomForestRegressor(n_estimators=50, max_depth=3).fit(data[treatment == 1][features], data[treatment == 1]['Y'])
            
            mu_0 = lambda x : y0_model.predict(x[features])  
            mu_1 = lambda x : y1_model.predict(x[features])  
        return RegressionWrapper(mu_0, mu_1)

# Decision tree model used for the experiment with misspecified estimators
class DecisionTreeModel:
    def fit_propensity(self, data, features):
        prop_model = DecisionTreeClassifier(max_depth=2)
        prop_model.fit(data[features], data['A'])

        return PropensityScoreWrapper(lambda x : prop_model.predict_proba(x)[:, 1])

    def fit_regression(self, data, features, binary = True):
        treatment = data['A']
        if binary: # Use classifier for predicting probabilities for binary data, regressor with scalar data
            y0_model = DecisionTreeClassifier(max_depth=2).fit(data[treatment == 0][features], data[treatment == 0]['Y']) 
            y1_model = DecisionTreeClassifier(max_depth=2).fit(data[treatment == 1][features], data[treatment == 1]['Y'])  

            mu_0 = lambda x : y0_model.predict_proba(x[features])[:, 1] 
            mu_1 = lambda x : y1_model.predict_proba(x[features])[:, 1]
        else:
            y0_model = DecisionTreeRegressor(max_depth=2).fit(data[treatment == 0][features], data[treatment == 0]['Y']) 
            y1_model = DecisionTreeRegressor(max_depth=2).fit(data[treatment == 1][features], data[treatment == 1]['Y'])  

            mu_0 = lambda x : y0_model.predict(x[features])  
            mu_1 = lambda x : y1_model.predict(x[features])  
        return RegressionWrapper(mu_0, mu_1)
