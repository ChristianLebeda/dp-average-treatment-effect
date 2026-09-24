import numpy as np
import pandas as pd
import math

# Generates covariates from N(0, 1) or Uniform([0,1])
def generate_synthetic_covariates(dim_x = 10, n = 1000, gaussian = True, scale = 1):
    #Generate data with "dim_x" covariates
    X_cols = ['X'+str(i) for i in range(1, dim_x + 1)] # + ['intercept']
    
    #Generate covariates
    data = pd.DataFrame(
        {'intercept': 1,
         'synthetic': 1, # Flag for synthetic data, this is used by oracle estimators
        **{x: np.random.normal(0, scale, n) if gaussian else np.random.uniform(0, scale, n) for x in X_cols},
    })

    return X_cols, data

def simulate_logistic_treatment_assignment(data, dim_x, X_cols, eta, beta = None):
    #Generate propensity score as a function of covariates
    def logistic_treatment(x, beta):
        return np.clip(1 / (1 + np.exp(-np.dot(x, beta))), eta, 1 - eta)
    if beta is None:
        # Generate random parameters
        beta = [np.random.uniform(-1, 1) for i in range(dim_x+1)]
    data['pi'] = logistic_treatment(data[['intercept'] + X_cols], beta)
    data['A'] = np.random.binomial(1, data['pi'])

# Simulations binary outcomes as 0 and 1
def simulate_logistic_outcomes(data, dim_x, X_cols, gamma, beta = None):
    #Generate outcome probability as a function of covariates and treatment
    def logistic_outcomes(x, beta, a):
        return 1 / (1 + np.exp(-np.dot(x, beta) - a * gamma))
    if beta is None:
        # Generate random parameters
        beta = [np.random.uniform(-1, 1) for i in range(dim_x+1)]
    probabilities0 = logistic_outcomes(data[['intercept'] + X_cols], beta, 0)
    probabilities1 = logistic_outcomes(data[['intercept'] + X_cols], beta, 1)
    data['Emu0'] = probabilities0
    data['Emu1'] = probabilities1
    data['mu0'] = np.random.binomial(1, probabilities0)
    data['mu1'] = np.random.binomial(1, probabilities1)
    # SUTVA
    data['Y'] = data['mu1'] * data['A'] + data['mu0'] * (1 - data['A'])
    # Return estimate of tau
    return (data['Emu1'].sum() - data['Emu0'].sum()) / data['intercept'].sum()

def generate_linear_outcomes(data, dim_x, X_cols, beta, tau, n, scale = 1, b = None):
    #Generate outcome. Covariate effect is simply a mean
    noise = np.random.normal(0, scale, n)
    data['Emu0'] = np.dot(data[['intercept'] + X_cols], beta)
    data['mu0'] = data['Emu0'] + noise
    data['Emu1'] = data['Emu0'] + tau
    data['mu1'] = data['mu0'] + tau
    
    if b: # Clip data to [-b,b]
        data['Emu0'] = np.clip(data['Emu0'], -b, b)
        data['mu0'] = np.clip(data['mu0'], -b, b)
        data['Emu1'] = np.clip(data['Emu1'], -b, b)
        data['mu1'] = np.clip(data['mu1'], -b, b)
    
    data['Y'] = data['mu1'] * data['A'] + data['mu0'] * (1 - data['A'])
    return (data['mu1'].sum() - data['mu0'].sum()) / data['intercept'].sum()

# Data with many propensity scores close to 0 and 1
def experiment1(n = 5000, eta = 0.004):
    dim_x = 1
    beta = [-0.2, 3.]
    beta2 = np.array([-0.05, 0.225])
    
    X_cols, data = generate_synthetic_covariates(dim_x, n, True)

    simulate_logistic_treatment_assignment(data, dim_x, X_cols, eta, beta)
    
    sample_tau = generate_linear_outcomes(data, dim_x, X_cols, beta2, 0.1, n, scale = 0.1, b=1)
    
    return X_cols, data, sample_tau

# Misspecified setting
# Region based propensity scores and outcomes
def experiment2(n = 250000, clip = True):
    dim_x = 2

    def thresholds1(X1, X2): # Used for propensity scores
        if X1 > 0.1 and X2 > 0:
            return 0.75
        if X1 <= 0.1 and X2 > 0:
            return 0.6
        if X1 < -0.05 and X2 <= 0:
            return 0.25
        return 0.5

    def thresholds2(X1, X2): # Used for expected outcomes
        if X1 > 0 and X2 > 0:
            return -0.7
        if X1 <= 0 and X2 > 0.05:
            return -0.4
        if X1 < 0 and X2 <= 0.05:
            return 0.6
        return 0.1
        
    X_cols, data = generate_synthetic_covariates(dim_x, n, True)

    # Define piecewise constant propensity score
    # Quadrant-based assignment
    data['pi'] = data.apply(
            lambda row: thresholds1(row['X1'], row['X2']),
            axis=1
        )
    data['A'] = np.random.binomial(1, data['pi'])
    

    tau = 0.2
    data['Emu0'] = data.apply(
            lambda row: thresholds2(row['X1'], row['X2']),
            axis=1
        )
    data['mu0'] = data['Emu0'] + np.random.normal(0, 0.05, n)
    data['Emu1'] = data['Emu0'] + tau
    data['mu1'] = data['mu0'] + tau

    if clip:
        # Alternative, if we want to map data from [-1,1] to [0,1]
        # data['Emu0'] = np.clip((data['Emu0'] + 1)/2, 0, 1) 
        # data['mu0'] = np.clip((data['mu0'] + 1)/2, 0, 1)
        # data['Emu1'] = np.clip((data['Emu1'] + 1)/2, 0, 1)
        # data['mu1'] = np.clip((data['mu1'] + 1)/2, 0, 1)
        data['Emu0'] = np.clip(data['Emu0'], -1, 1)
        data['mu0'] = np.clip(data['mu0'], -1, 1)
        data['Emu1'] = np.clip(data['Emu1'], -1, 1)
        data['mu1'] = np.clip(data['mu1'], -1, 1)

    data['Y'] = data['mu1'] * data['A'] + data['mu0'] * (1 - data['A']) # SUTVA
    sample_tau = (data['mu1'].sum() - data['mu0'].sum()) / data['intercept'].sum()
    
    #print("Sample treatment effect", sample_tau) # Sanity check to ensure that clipping did not mess up the treatment effect
    
    return X_cols, data, sample_tau

# Generate data friendly for previous work.
# Logistic propensity score and outcomes
def experiment3(n = 50000, eta = 0.1):
    dim_x = 10
    beta1 = [0.1, -0.15, 0.225, -0.15, -0.2, 0.1, 0.05, -0.075, 0.225, -0.15, -0.2] 
    beta2 = [-0.05, 0.175, 0.1, -0.125, 0.075, -0.1, 0.2, -0.2, 0.175, -0.1, 0.2] 

    X_cols, data = generate_synthetic_covariates(dim_x, n, True)

    simulate_logistic_treatment_assignment(data, dim_x, X_cols, eta, beta1)
    
    sample_tau =  simulate_logistic_outcomes(data, dim_x, X_cols, 0.42585, beta2)
    
    return X_cols, data, sample_tau


# Generate data for well-specified setting for changing K
# Logistic propensity score and outcomes
def experiment4(n = 20000, eta = 0.1):
    dim_x = 20
    beta1 = [0.1, -0.17, -0.06, 0.05, 0.14, 0.12, -0.195, -0.205, 0.07, 0.18, 0.14, -0.14, 0.05, 0.01, -0.16, -0.18, -0.1, 0.2, 0.03, -0.16, -0.1]
    beta2 = np.array([-0.08,-0.0385, -0.0111, -0.105, -0.0344,  0.1405,
        0.0550,  0.0344, -0.0908, -0.0023, -0.0243,
       -0.0076, -0.0416,  0.0193, -0.0846,  0.0582,
        0.0824,  0.0184,  0.0064, -0.0895,  0.0241])

    X_cols, data = generate_synthetic_covariates(dim_x, n, True)

    simulate_logistic_treatment_assignment(data, dim_x, X_cols, eta, beta1)
    
    sample_tau =  generate_linear_outcomes(data, dim_x, X_cols, beta2, 0.15, n, scale = 0.05, b=1)
    
    return X_cols, data, sample_tau


def generate_experiment1_files(n = 5000, eta = 0.004, repetitions = 10):
    # Fix seed for data generation
    np.random.seed(42)
    
    for i in range(repetitions):
        X_cols, data, sample_tau = experiment1(n, eta)

        # Save in multiple formats
        data.to_pickle(f"data/experiment1-{i}")

def generate_experiment2_files(n = 250000, repetitions = 10):
    # Fix seed for data generation
    np.random.seed(42)
    
    for i in range(repetitions):
        X_cols, data, sample_tau = experiment2(n, eta)

        # Save in multiple formats
        data.to_pickle(f"data/experiment2-{i}")

def generate_experiment3_files(n = 50000, eta = 0.1, repetitions = 10):
    # Fix seed for data generation
    np.random.seed(42)
    
    for i in range(repetitions):
        X_cols, data, sample_tau = experiment3(n, eta)

        # Save in multiple formats
        data.to_pickle(f"data/experiment3-{i}")

def generate_experiment4_files(n = 20000, eta = 0.1, repetitions = 10):
    # Fix seed for data generation
    np.random.seed(42)
    
    for i in range(repetitions):
        X_cols, data, sample_tau = experiment4(n, eta)

        # Save in multiple formats
        data.to_pickle(f"data/experiment4-{i}")

if __name__ == "__main__":
    print("Example data")
    print(experiment1(5000, 0.004))
    print(experiment2(250000, 0.2))
    print(experiment3(50000, 0.1))
    print(experiment4(20000, 0.1))
    #generate_experiment1_files(50000, repetitions = 100)
