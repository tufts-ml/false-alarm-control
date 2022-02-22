import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler

from ap_perf import PerformanceMetric, MetricLayer

np.random.seed(0)

def create_toy_data():
    #create toy data
    # generate 225 positive data points
    n_P = [30, 60, 30]
    # n_P = [60, 120, 60]
    P = 3
    prng = np.random.RandomState(101)

    # the first 25 data points come from mean 2-D mvn with mean [1, 2.5] and next 200 come from
    # 2-D mvn with mean [1, 1]
    mu_PD = np.asarray([
        [0.7, 2.5],
        [0.7, 1.0],
        [0.7, 0.0]])

    cov_PDD = np.vstack([
        np.diag([0.06, 0.1])[np.newaxis,:],
        np.diag([0.1, 0.1])[np.newaxis,:],
        np.diag([0.06, 0.06])[np.newaxis,:],
        ])

    xpos_list = list()
    for p in range(P):
        x_ND = prng.multivariate_normal(mu_PD[p], cov_PDD[p], size=n_P[p])
        xpos_list.append(x_ND)
    x_pos_ND = np.vstack(xpos_list)
    y_pos_N  = np.ones(x_pos_ND.shape[0])

    # generate 340 negative data points
    n_P = [400, 30, 20]
    # n_P = [800, 60, 40]
    P = 3
    prng = np.random.RandomState(201)

    # the first 300 data points come from mean 2-D mvn with mean [2.2, 1.5] and next 20 come from
    # 2-D mvn with mean [0, 3] and next 20 from 2-D mvn with mean [0, 0.5]
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

    xneg_list = list()
    for p in range(P):
        x_ND = prng.multivariate_normal(mu_PD[p], cov_PDD[p], size=n_P[p])
        xneg_list.append(x_ND)
    x_neg_ND = np.vstack(xneg_list)
    y_neg_N = np.zeros(x_neg_ND.shape[0])

    x_ND = np.vstack([x_pos_ND, x_neg_ND])
    y_N = np.hstack([y_pos_N, y_neg_N])

    x_ND = (x_ND - np.mean(x_ND, axis=0))/np.std(x_ND, axis=0)

    x_pos_ND = x_ND[y_N == 1]
    x_neg_ND = x_ND[y_N == 0]

    return x_ND, y_N


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx, :], self.y[idx]]

    
class LinearClassifier(nn.Module):
    def __init__(self, nvar):
        super().__init__()
        self.fc1 = nn.Linear(nvar, 1)
        self.double()
        
    def forward(self, x):
        x = self.fc1(x)
        return x.squeeze()



class RecallGvPrecision(PerformanceMetric):
    def __init__(self, th):
        self.th = th

    def define(self, C):
        return C.tp / C.ap  

    def constraint(self, C):
        return C.tp / C.pp  >= self.th   

    
if __name__=='__main__':
    precision_gv_recall_80 = PrecisionGvRecall(0.8)
    precision_gv_recall_80.initialize()
    precision_gv_recall_80.enforce_special_case_positive()
    precision_gv_recall_80.set_cs_special_case_positive(True)

    recall_gv_precision_80 = RecallGvPrecision(0.8)
    recall_gv_precision_80.initialize()
    recall_gv_precision_80.enforce_special_case_positive()
    recall_gv_precision_80.set_cs_special_case_positive(True)


    # performance metric
    pm = recall_gv_precision_80


    X_tr, y_tr = create_toy_data()

    trainset = TabularDataset(X_tr, y_tr)
    # testset = TabularDataset(X_ts, y_ts)


    batch_size = len(X_tr) # full batch
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    method = "ap-perf"              # uncomment if we want to use ap-perf objective 
    # method = "bce-loss"           # uncomment if we want to use bce-loss objective

    torch.manual_seed(189)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    nvar = X_tr.shape[1]
    model = LR(nvar).to(device)

    if method == "ap-perf":
        criterion = MetricLayer(precision_gv_recall_80).to(device)
        lr = 3e-3
        weight_decay = 0
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
        lr = 1e-2
        weight_decay = 1e-3

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(2): # epoch 2 was the best checkpoint after running 50 epochs

        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            if epoch>=2:
                optimizer.param_groups[0]['lr'] = 0.1*optimizer.param_groups[0]['lr']

            sys.stdout.write("\r#%d | progress : %d%%" % (epoch,  int(100 * (i+1) / len(trainloader))))
            sys.stdout.flush()

        print()

        # evaluate after each epoch
        model.eval()

        # train
        train_data = torch.tensor(X_tr).to(device)
        tr_output = model(train_data)
        tr_pred = (tr_output >= 0.0).float()
        tr_pred_np = tr_pred.cpu().numpy()

        train_acc = np.sum(y_tr == tr_pred_np) / len(y_tr)
        train_metric = pm.compute_metric(tr_pred_np, y_tr)
        train_constraint = pm.compute_constraints(tr_pred_np, y_tr)


        model.train()

        print('#{} | Acc tr: {:.5f} | Metric tr: {:.5f} | Constraint tr: {:.5f}'.format(
            epoch, train_acc, train_metric, train_constraint[0]))
    
