import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import precision_score
from sklearn.svm import SVR
from random import sample 
import numpy as np
import warnings
from sklearn.kernel_approximation import Nystroem

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

import sys
import os
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from fastFM import sgd
import scipy.sparse as sp
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
import csv


start = time.time()


SK = 128
GN_OM = pd.read_csv('data/test data/dendritic_Gene_Ori_Map_%d.tsv' %SK,'\t',header = None)


GN_Y = []
s = open("dendritic/gold_standard_dendritic_whole.txt")  # 'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
for line in s:
    separation = line.split()
    GN_Y.append(separation[2])
    

GN_Y = np.array(GN_Y)
GN_OY = GN_Y
print(GN_OY.shape)



GN_OM = np.array(GN_OM.values.tolist())


print(GN_OM.shape)

#GN_OM = GN_OM/1000000

GN_OY = GN_Y.astype(np.int)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

avr_auroc_l = [[],[],[],[]]
avr_aupr_l = [[],[],[],[]]

for train_index, test_index in skf.split(GN_OM,GN_OY):

    GN_Mat = GN_OM[train_index]

    GN_Y = GN_OY[train_index]

    
    GN_MT = GN_Mat[GN_Y == 1]
    GN_MF = GN_Mat[GN_Y == 0]

    GN_Mat = np.vstack((GN_MT, GN_MF))
    GN_Y = np.hstack((np.ones(len(GN_MT),dtype = int), np.zeros(len(GN_MF),dtype = int)))
    
    
    print(GN_Y.shape)

    
    GN_TMat = GN_OM[test_index]
    GN_TY = GN_OY[test_index]

    print(GN_TY.shape)
    
    print(GN_TMat.shape)
    
    
    X_train = sp.csc_matrix(GN_Mat, dtype=np.float64)
    X_test = sp.csc_matrix(GN_TMat, dtype=np.float64)
    GN_B = np.where(GN_Y == 1, 1, -1)
    y_train = np.array(GN_B)
    
    k = 8
    for i in range(1):
      k *= 2
      fm = sgd.FMClassification(n_iter=len(GN_Mat)*10, step_size = 0.1, n_iter1=0, init_stdev=0.1, rank=k, l2_reg_w=0.0, l2_reg_V=0.0, is_w = 1, pr = 0)
      fm.fit(X_train, y_train)
      y_pred = fm.predict(X_test)
      print(k)
      auroc = roc_auc_score(GN_TY,y_pred)
      aupr = average_precision_score(GN_TY,y_pred)

      print("auroc:",auroc)
      print("aupr:",aupr)
      avr_auroc_l[i].append(auroc)
      avr_aupr_l[i].append(aupr)



#print("HSIC ","auroc:", avr_auroc, "aupr:", avr_aupr)

for i in range(1):
  print("FM","low_rank=", 2**(i+3), "auroc:%.4f$\\pm$%.2f" %(np.mean(avr_auroc_l[i]), np.std(avr_auroc_l[i], ddof=1)),\
  "aupr:%.4f$\\pm$%.2f" %(np.mean(avr_aupr_l[i]), np.std(avr_aupr_l[i], ddof=1)))

end = time.time()
print("time:", end-start)
  
  


