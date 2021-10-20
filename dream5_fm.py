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
from sklearn.metrics import precision_recall_curve
from fastFM import sgd
import scipy.sparse as sp
import time
from sklearn.model_selection import KFold


GN_OM = pd.read_csv('data/test data/DREAM5_NetworkInference_Network2_GK_Gene_Ori_Map2.tsv','\t',header = None)
hsic = pd.read_csv('data/test data/DREAM5_NetworkInference_Network2_GK_HSIC2.tsv','\t',header = None)

hsic = np.array(hsic.values.tolist())
GN_OM = np.array(GN_OM.values.tolist())

print(type(GN_OM[0][0]))
print(GN_OM[0])
GN_OM = GN_OM/5000000
#GN_OM = GN_OM/50

Half = len(hsic)//2
print(Half)
GN_OY = np.hstack((np.ones(Half,dtype = int), np.zeros(Half,dtype = int)))
index = [i for i in range(Half*2)]

start = time.time()
lenths = [50, 100, 200, 500]
AUROC = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
AUPR = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
for j in range(5):
  for kk in range(len(lenths)):
    lenth = lenths[kk]
    print("Length =", lenth)
    avr_auroc = 0.0
    avr_aupr = 0.0
    avr_auroc_l = [0.0,0.0,0.0,0.0]
    avr_aupr_l = [0.0,0.0,0.0,0.0]
    for iter in range(5):
      train_index = sample(index, lenth)
      test_index = [i for i in index if i not in train_index]
      
      GN_Mat = GN_OM[train_index]
      GN_Y = GN_OY[train_index]
      
      GN_TMat = GN_OM[test_index] 
      GN_TY = GN_OY[test_index]
      HSIC = hsic[test_index]
      
      auroc = roc_auc_score(GN_TY,HSIC)
      aupr = average_precision_score(GN_TY,HSIC)
      #print("auroc:",auroc)
      #print("aupr:",aupr)
      avr_auroc += auroc
      avr_aupr += aupr
      
      X_train = sp.csc_matrix(GN_Mat, dtype=np.float64)
      X_test = sp.csc_matrix(GN_TMat, dtype=np.float64)
      GN_B = np.where(GN_Y == 1, 1, -1)
      y_train = np.array(GN_B)
      #print(y_train)
      
      k = 8
      for i in range(1):
        k *= 2
        fm = sgd.FMClassification(n_iter=lenth*10, n_iter1=0, init_stdev=0.1, rank=k, l2_reg_w=0.0, l2_reg_V=0.0, is_w = 1)
        fm.fit(X_train, y_train)
        y_pred = fm.predict(X_test)
        print(k)
        #print(y_pred)
        #print(fm.w_)
        #print(fm.V_)
        auroc = roc_auc_score(GN_TY,y_pred)
        aupr = average_precision_score(GN_TY,y_pred)
        print("auroc:",auroc)
        print("aupr:",aupr)
        avr_auroc_l[i] += auroc
        avr_aupr_l[i] += aupr
    
    print("Length =", lenth)
    print("HSIC ","auroc:", avr_auroc/5, "aupr:", avr_aupr/5)
    for i in range(4):
      print("FM","low_rank=", 2**(i+1), "auroc:", avr_auroc_l[i]/5, "aupr:", avr_aupr_l[i]/5)
      AUROC[kk][i].append(avr_auroc_l[i]/5)
      AUPR[kk][i].append(avr_aupr_l[i]/5)
      
for k in range(len(lenths)):
  for i in range(4):
    print("FM",lenths[k],"low_rank=", 2**(i+1), "auroc:%.2f$\\pm$%.2f" %(np.mean(AUROC[k][i]), np.std(AUROC[k][i], ddof=1)),\
    "aupr:%.2f$\\pm$%.2f" %(np.mean(AUPR[k][i]), np.std(AUPR[k][i], ddof=1)))
  
end = time.time()
print("Time",end-start)


