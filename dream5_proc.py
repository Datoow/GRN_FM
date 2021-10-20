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

from sklearn.model_selection import KFold

#,header=None
Gold_Net = pd.read_csv('data/test data/DREAM5_NetworkInference_GoldStandard_Network2.tsv',',',header=None)
#predict_net = pd.read_csv('DREAM5_NetworkInference_HSICLasso_Network1',',')
#predict_net = pd.read_csv('data/test data/DREAM5_NetworkInference_HSICLasso_Network2',',')
#predict_net = pd.read_csv('data/test data/Only_edge/DREAM5_predict_HSICLasso_Network2_Chain.tsv','\t')
Half = 518 #int(input("Half?"))

Gold_Net = Gold_Net.values
#np.random.shuffle(Gold_Net)

GN_X = Gold_Net[:,0:2]
GN_Y = Gold_Net[:,2]

if len(GN_Y[GN_Y==2]) > 0:
  GN_X = GN_X[1:]
  GN_Y = GN_Y[1:]
#for i in GN_OY:
#  print(i)

GN_T = GN_X[:Half]
GN_F = GN_X[Half:]

np.random.shuffle(GN_F)
GN_F = GN_F[:Half]
'''
GN_OT = GN_X[GN_Y == 1]
GN_OF = GN_X[GN_Y == 0]
Half = len(GN_OT)
GN_T = GN_OT
GN_F = GN_OF[:Half]
'''
GN_OX = []
GN_OY = []
for i in range(len(GN_T)):
  GN_OX.append(GN_T[i])
  GN_OX.append(GN_F[i])
  GN_OY.append(1)
  GN_OY.append(0)
GN_OX = np.array(GN_OX)
GN_OY = np.array(GN_OY)

GN_OX = np.vstack((GN_T, GN_F))
GN_OY = np.vstack((np.ones((Half,1)), np.zeros((Half,1))))

print(GN_OX.shape)
print(GN_OY.shape)

#index = [i for i in range(Half*2)]
SK=16

kernel_data = pd.read_csv('data/test data/DREAM5_NetworkInference_Network2_GK.tsv','\t')
#print(kernel_data.columns)
#coldstress_data = coldstress_data.sample(n=10,random_state=123,axis=0)

#feature_map_nystroem = Nystroem(gamma=.2, random_state=1, n_components=50)
#print(len(coldstress_data[GN_X[1]]))
avr_auroc = 0.0
avr_aupr = 0.0

hsic = np.zeros((1,1))
#GN_OM = np.zeros((1,2*16384))
GN_OM = np.zeros((1,2*SK*SK))

for i in range(len(GN_OX)):
  exp_data = np.array(kernel_data[GN_OX[i]].values.tolist())
  #print(exp_data[:,0].shape)
  #print(exp_data[:,1].shape)
  GN_OM = np.vstack((GN_OM, np.hstack((exp_data[:,0].T,exp_data[:,1].T))))
  hsic = np.vstack((hsic, sum(exp_data[:,0].T*exp_data[:,1].T)))
  
GN_OM = GN_OM[1:]
hsic = hsic[1:]

print(GN_OM.shape)
print(hsic.shape)

pr1 = pd.DataFrame(GN_OM)
pr1.to_csv('data/test data/DREAM5_NetworkInference_Network2_GK_Gene_Ori_Map2.tsv','\t',index=False,header=None)

pr2 = pd.DataFrame(hsic)
pr2.to_csv('data/test data/DREAM5_NetworkInference_Network2_GK_HSIC2.tsv','\t',index=False,header=None)