import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
from theano import tensor as T
import pandas
import itertools

from construct_data import construct_data



def sample_quasi_posterior(data, data_test, Y, C, beta0_prior, beta1_prior, alpha_prior,
                           n_samples, burn, thin=20):

    with pm.Model() as exponential_quasi_bayesian:
        #Regressors
        if beta0_prior[0] == 'Uniform':
            beta0 = pm.Uniform('beta0', beta0_prior[1], beta0_prior[2])
        elif beta0_prior[0] == 'Gamma':
            beta0 = pm.Gamma('beta0', beta0_prior[1], beta0_prior[2])
        elif beta0_prior[0] == 'Normal':
            beta0 = pm.Normal('beta0', beta0_prior[1], beta0_prior[2])
        else:
            beta0 = None
            print("Error beta0 prior type not in 'Uniform', 'Gamma', 'Normal'")
            
        if beta1_prior[0] == 'Uniform':
            beta1 = pm.Uniform('beta1', beta1_prior[1], beta1_prior[2])
        elif beta1_prior[0] == 'Gamma':
            beta1 = pm.Gamma('beta1', beta1_prior[1], beta1_prior[2])
        elif beta1_prior[0] == 'Normal':
            beta1 = pm.Normal('beta1', beta1_prior[1], beta1_prior[2])
        else:
            beta1 = None
            print("Error beta1 prior type not in 'Uniform', 'Gamma', 'Normal'")
        
        if alpha_prior[0] == 'Uniform':
            alpha = pm.Uniform('alpha', alpha_prior[1], alpha_prior[2])
        elif alpha_prior[0] == 'Gamma':
            alpha = pm.Gamma('alpha', alpha_prior[1], alpha_prior[2])
        elif alpha_prior[0] == 'Normal':
            alpha = pm.Normal('alpha', alpha_prior[1], alpha_prior[2])
        elif alpha_prior[0] == 'Constant':
            alpha = alpha_prior[1]
        else:
            alpha = None
            print("Error alpha prior type not in 'Uniform', 'Gamma', 'Normal', 'Constant'")
        
        
        lambda_ = pm.Deterministic('lambda_', T.exp(beta0 * data.a + beta1*data.b))
        #Prediction
        y_hat = pm.Deterministic('y_hat', (1 - T.exp(-C*lambda_))/lambda_)
 
        def log_exp_risk(failure):
            out = ((y_hat - failure)**2).mean()
            return -alpha*out
        
        #Posterior Predictive Distribution
        lambda_test = pm.Deterministic('lambda_test', T.exp(beta0*data_test.a + beta1*data_test.b))
        y_pred = pm.Deterministic('y_pred', (1 - T.exp(-C*lambda_test))/lambda_test)
        
        exp_surv = pm.DensityDist('exp_surv', log_exp_risk, observed={'failure':Y.time})
    
    # --SAMPLES --
    with exponential_quasi_bayesian:
        step = pm.Metropolis()
        trace = pm.sample(n_samples, step)

    trace = trace[burn::thin]
    return trace

def run_several_expo(n_iter, beta_true, size_data_list, C, beta0_prior_list,
                     beta1_prior_list, alpha_prior_list, n_samples, burn, plot, thin=20):
    
    prior_grid = list(itertools.product(*[beta0_prior_list, beta1_prior_list, alpha_prior_list]))
    
    param_results = {'n_iter': [], 'N': [], 'beta0_prior': [], 'beta1_prior': [],
                     'alpha_prior': [], 'beta0_MQP': [], 'beta1_MQP': [], 'mse':[]}
    
    for N in size_data_list:
        data, Y, delta_list = construct_data(beta_true, N, C)
        data = pandas.DataFrame(data)
        data = data.rename(index=str, columns={0: "a", 1: "b"})
        
        data_test, Y_test, delta_list_test = construct_data(beta_true, 2000, C)
        data_test = pandas.DataFrame(data_test)
        data_test = data.rename(index=str, columns={0: "a", 1: "b"})
        
        Y = pandas.DataFrame(Y)
        Y = Y.rename(index=str, columns={0: "time"}) #death time

        delta_list = np.array(delta_list)
        
        for beta0_prior, beta1_prior, alpha_prior in prior_grid:
            for i in range(n_iter):
                trace = sample_quasi_posterior(data, data_test, Y, C, beta0_prior, beta1_prior, alpha_prior,
                               n_samples, burn, thin)
                
                pm.traceplot(trace);
                plt.savefig("N: {}, beta0_prior: {}, beta1_prior: {}, alpha_prior: {}_{}".format(
                    str(N), str(beta0_prior), str(beta1_prior),  str(alpha_prior), i), format="png")
                
                #Posterior Prediction
                Y_pred = trace["y_pred"].mean(axis=0)
                mse = ((Y_test - Y_pred)**2).mean()
                
                if plot:
                    print("beta0 mean a quasi posteriori : ", trace['beta0'].mean())
                    print("beta1 mean a quasi posteriori : ", trace['beta1'].mean())
                    print("Mean Squared Error: %0.2f" % mse)
                    
                param_results['n_iter'].append(i)
                param_results['N'].append(N)
                param_results['beta0_prior'].append(beta0_prior)
                param_results['beta1_prior'].append(beta1_prior)
                param_results['alpha_prior'].append(alpha_prior)
                param_results['beta0_MQP'].append(trace['beta0'].mean())
                param_results['beta1_MQP'].append(trace['beta1'].mean())
                param_results['mse'].append(mse)
    return pandas.DataFrame(param_results)