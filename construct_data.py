import numpy as np

def construct_data(beta_true, N, c):

    ages = [np.random.randint(18, 80) for n in range(N)]
    levels = [np.random.random() for n in range(N)]
    data = np.array([ages, levels]).T

    y_list = []
    delta_list = []
    for i in range(N):
        lambda_i = np.exp(beta_true.dot(data[i, :]))
        death = np.random.exponential(1/lambda_i)
        if death > c:
            y_list.append(c)
            delta_list.append(1)
        else:
            y_list.append(round(death, 0) + 1)
            delta_list.append(0)
    return data, np.array(y_list), delta_list

def quasi_posterior(beta, X, Y, c, alpha=0.01):
    """ Compute the quasi posterior i.e exp(-alpha * risk) with
        - alpha: temperature of the model
        - risk: least square risk 
        - X: data
        - Y: ground truth, here the time of survival
        - c: right censure"""
    
    n = len(X)
    lambda_list = [np.exp(X[i, :].dot(beta)) for i in range(n)]
    Y_hat = [(1 - np.exp(-lambda_list[i]*c))/ lambda_list[i] for i in range(n)]
    risk = np.sum([(Y_hat[i] - Y[i])**2 for i in range(n)])
    return np.exp(-alpha/n * risk)