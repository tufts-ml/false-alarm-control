import numpy as np

def calc_decision_score(x_ND, w_D):
    # Compute real-valued scores fed into sigmoid
    u_N = np.dot(x_ND, w_D[:-1]) + w_D[-1]
    return u_N

def calc_binary_clf_perf(y_N, yhat_N, gamma=None, delta=None):
    TP = np.sum(np.logical_and(y_N == 1, yhat_N == 1))
    FP = np.sum(np.logical_and(y_N == 0, yhat_N == 1))
    TN = np.sum(np.logical_and(y_N == 0, yhat_N == 0))
    FN = np.sum(np.logical_and(y_N == 1, yhat_N == 0))
    assert np.allclose(y_N.size, TP+FP+TN+FN)
    precision = TP / (1e-10 + float(TP + FP))
    recall = TP / (1e-10 + float(TP + FN))
    perfdict = dict(
        precision=precision,
        recall=recall,
        TP=TP,
        TN=TN,
        FP=FP,
        FN=FN)
    if delta is not None:
    	perfdict['TP+gamma*delta'] = TP + gamma * delta * (TP+FN)
    return perfdict