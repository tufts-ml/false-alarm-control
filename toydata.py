import numpy as np
import pandas as pd
import scipy.stats

def create_toy_dataset(
        pos_random_state=101,
        neg_random_state=201,
        ):
    ''' Generate data from distinct Gaussian blobs

    3 different blobs are "positive"
    3 different blobs are "negative"

    Returns
    -------
    x_NF : 2D array, shape (N, F) = (n_examples, n_features)
    y_N  : 2D array, shape (N,)
    '''


    # Generate positive data from 3 blobs
    P = 3
    n_P = [30, 60, 30]    
    mu_PD = np.asarray([
        [0.7, 2.5],
        [0.7, 1.0],
        [0.7, 0.0]])
    cov_PDD = np.vstack([
        np.diag([0.06, 0.1])[np.newaxis,:],
        np.diag([0.1, 0.1])[np.newaxis,:],
        np.diag([0.06, 0.06])[np.newaxis,:],
        ])

    prng = np.random.RandomState(int(pos_random_state))
    xpos_list = list()
    for p in range(P):
        x_ND = prng.multivariate_normal(mu_PD[p], cov_PDD[p], size=n_P[p])
        xpos_list.append(x_ND)
    x_pos_ND = np.vstack(xpos_list)
    y_pos_N  = np.ones(x_pos_ND.shape[0])

    # Generate negative data from 3 blobs
    P = 3
    n_P = [400, 30, 20]
    mu_PD = np.asarray([
        [2.25, 1.5],
        [0.0, 3.0],
        [0.0, 0.5],
        ])
    cov_PDD = np.vstack([
        np.diag([.1, .2])[np.newaxis,:],
        np.diag([.05, .05])[np.newaxis,:],
        np.diag([.05, .05])[np.newaxis,:],
        ])

    prng = np.random.RandomState(int(neg_random_state))
    xneg_list = list()
    for p in range(P):
        x_ND = prng.multivariate_normal(mu_PD[p], cov_PDD[p], size=n_P[p])
        xneg_list.append(x_ND)
    x_neg_ND = np.vstack(xneg_list)
    y_neg_N = np.zeros(x_neg_ND.shape[0])

    # Aggregate the positive and negative data together
    x_ND = np.vstack([x_pos_ND, x_neg_ND])
    y_N = np.hstack([y_pos_N, y_neg_N])

    # Standardize the data
    x_ND = (x_ND - np.mean(x_ND, axis=0))/np.std(x_ND, axis=0)
    x_pos_ND = x_ND[y_N == 1]
    x_neg_ND = x_ND[y_N == 0]

    return x_ND, y_N, x_pos_ND, y_pos_N, x_neg_ND, y_neg_N

def create_toy_dataset_large(
        pos_random_state=101,
        neg_random_state=201,
        ):
    ''' Generate data from distinct Gaussian blobs

    3 different blobs are "positive"
    3 different blobs are "negative"

    Returns
    -------
    x_NF : 2D array, shape (N, F) = (n_examples, n_features)
    y_N  : 2D array, shape (N,)
    '''


    # Generate positive data from 3 blobs
    P = 3
    n_P = [5000, 10000, 5000]    
    mu_PD = np.asarray([
        [0.7, 2.5],
        [0.7, 1.0],
        [0.7, -0.5]])
    cov_PDD = np.vstack([
        np.diag([0.06, 0.1])[np.newaxis,:],
        np.diag([0.1, 0.1])[np.newaxis,:],
        np.diag([0.06, 0.1])[np.newaxis,:],
        ])

    prng = np.random.RandomState(int(pos_random_state))
    xpos_list = list()
    for p in range(P):
        x_ND = prng.multivariate_normal(mu_PD[p], cov_PDD[p], size=n_P[p])
        xpos_list.append(x_ND)
    x_pos_ND = np.vstack(xpos_list)
    y_pos_N  = np.ones(x_pos_ND.shape[0])

    # Generate negative data from 3 blobs
    P = 3
    n_P = [350000, 35000, 20000]
    mu_PD = np.asarray([
        [2.25, 1.5],
        [0.0, 3.0],
        [0.0, 0.5],
        ])
    cov_PDD = np.vstack([
        np.diag([.1, .2])[np.newaxis,:],
        np.diag([.05, .05])[np.newaxis,:],
        np.diag([.05, .05])[np.newaxis,:],
        ])

    prng = np.random.RandomState(int(neg_random_state))
    xneg_list = list()
    for p in range(P):
        x_ND = prng.multivariate_normal(mu_PD[p], cov_PDD[p], size=n_P[p])
        xneg_list.append(x_ND)
    x_neg_ND = np.vstack(xneg_list)
    y_neg_N = np.zeros(x_neg_ND.shape[0])

    # Aggregate the positive and negative data together
    x_ND = np.vstack([x_pos_ND, x_neg_ND])
    y_N = np.hstack([y_pos_N, y_neg_N])

    # Standardize the data
    x_ND = (x_ND - np.mean(x_ND, axis=0))/np.std(x_ND, axis=0)
    x_pos_ND = x_ND[y_N == 1]
    x_neg_ND = x_ND[y_N == 0]

    return x_ND, y_N, x_pos_ND, y_pos_N, x_neg_ND, y_neg_N