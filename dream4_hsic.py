import pandas as pd
import numpy.matlib
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.preprocessing import normalize
import csv
from scipy.stats import pearsonr

from sklearn.metrics import precision_score
from sklearn.svm import SVR
from random import sample 
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


num = 5
coldstress_data = pd.read_csv('data/test data/insilico_size100_%d_dream4_timeseries.tsv' %num,'\t')
#coldstress_data = coldstress_data.drop([0], axis=1)
#coldstress_data = coldstress_data.sample(n=300,random_state=123,axis=0)

cols = coldstress_data.columns
#print(cols)
def kernel_gaussian(X_in_1, X_in_2, sigma):
    X_in_1 = X_in_1.T
    X_in_2 = X_in_2.T
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    X_in_12 = np.sum(np.power(X_in_1, 2), 0)
    #print(X_in_12)
    X_in_22 = np.sum(np.power(X_in_2, 2), 0)
    #print(X_in_22)
    dist_2 = np.tile(X_in_22, (n_1, 1)) + \
        np.tile(X_in_12, (n_2, 1)).transpose() - 2 * np.dot(X_in_1.T, X_in_2)
    sigma = np.sqrt(np.average(dist_2)) * (2 ** (0))
    #print(dist_2)
    K = np.exp(-dist_2 / (2 * np.power(sigma, 2)))
    return K

H = np.matlib.identity(210, dtype = float) - 1/210.0 * np.ones((210,210))

for i in range(1, len(cols)):
  exp = np.array(coldstress_data[[cols[i]]].values.tolist())

  #print(exp)
  kel = kernel_gaussian(exp, exp, np.sqrt(len(exp)))
  #print(kel)
  kel = kel / (np.linalg.norm(kel, 'fro') + 10e-10)
  
  Ht = H.copy()
  
  Ht = H @ kel
  
  kel = Ht @ H
  
  kel = kel.flatten()
  
  kel = kel.reshape(1,-1)
  #print(kel)
  kel=normalize(kel, norm = 'max')
  
  kel2 = kel.copy()
  
  #print(kel)
  kel = np.uint16(kel*65535)
  #print(kel)
  exp=exp.T
  #kel = kel.T
  if i == 1:
    E = kel
    E2 = exp
    E3 = kel2
  else:
    E = np.vstack((E,kel))
    E2 = np.vstack((E2,exp))
    E3 = np.vstack((E3,kel2))

E = E.T
E2 = E2.T
E3 = E3.T
print(E.shape)
print(E2.shape)
print(E3.shape)

kernel_data = pd.DataFrame(E)
kernel_data.columns = cols[1:]

exp_data = pd.DataFrame(E2)
exp_data.columns = cols[1:]

kernel_data2 = pd.DataFrame(E3)
kernel_data2.columns = cols[1:]

#,header=None
#Gold_Net = pd.read_csv('data/test data/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv','\t',header=None)
#num = int(input("num?"))
Gold_Net = pd.read_csv('data/test data/insilico_size100_%d_goldstandard.tsv' %num,'\t',header=None)

Gold_Net = Gold_Net.values
#np.random.shuffle(Gold_Net)

auroc_PC = []
aupr_PC = []
auroc_HSIC = []
aupr_HSIC = []
auroc_FM = []
aupr_FM = []

for j in range(1):
  GN_X = Gold_Net[:,0:2]
  GN_Y = Gold_Net[:,2]
  
  Half = np.sum(GN_Y == 1)
  print("Half:", Half)
  
  if len(GN_Y[GN_Y==2]) > 0:
    GN_X = GN_X[1:]
    GN_Y = GN_Y[1:]
  #for i in GN_OY:
  #  print(i)
  
  GN_T = GN_X[:Half]
  GN_F = GN_X[Half:]
  
  #np.random.shuffle(GN_F)
  
  #GN_F = GN_F[:Half]
  
  pr = pd.DataFrame(GN_F) 
  #pr.to_csv('data/test data/DREAM5_NetworkInference_Net%d_%d.tsv' %(num, j),'\t',index=False)
  
  GN_OX = np.vstack((GN_T, GN_F))
  GN_OY = np.hstack((np.ones(len(GN_T),dtype = int), np.zeros(len(GN_F),dtype = int)))
  
  GN_OY = GN_OY.T
  
  print(GN_OX.shape)
  print(GN_OY.shape)
  
  
  avr_auroc = 0.0
  avr_aupr = 0.0
  
  
  for i in range(len(GN_OX)):
    k_data = np.array(kernel_data[GN_OX[i]].values.tolist())
    e_data = np.array(exp_data[GN_OX[i]].values.tolist())
    k_data2 = np.array(kernel_data2[GN_OX[i]].values.tolist())
  
    if i == 0:
      #GN_OM = np.hstack((k_data[:,0].T,k_data[:,1].T))
      hsic = np.array([sum(k_data2[:,0].T*k_data2[:,1].T/1e9)])
      #pc = np.array([abs(pearsonr(e_data[:,0].T,e_data[:,1].T)[0])])
    else:
      #GN_OM = np.vstack((GN_OM,np.hstack((k_data[:,0].T,k_data[:,1].T))))
      hsic = np.vstack((hsic,np.array([sum(k_data2[:,0].T*k_data2[:,1].T/1e9)])))
      #pc = np.vstack((pc,np.array([abs(pearsonr(e_data[:,0].T,e_data[:,1].T)[0])])))
  
    
  
  print("HSIC")
  avr_auroc = roc_auc_score(GN_OY,hsic)
  avr_aupr = average_precision_score(GN_OY,hsic)
  print("auroc:",avr_auroc)
  print("aupr:",avr_aupr)
  auroc_HSIC.append(avr_auroc)
  aupr_HSIC.append(avr_aupr)
  
  
#print("PC", "auroc:%.2f$\\pm$%.2f" %(np.mean(auroc_PC), np.std(auroc_PC, ddof=1)),\
  #  "aupr:%.2f$\\pm$%.2f" %(np.mean(aupr_PC), np.std(aupr_PC, ddof=1)))
print("HSIC", "auroc:%.2f$\\pm$%.2f" %(np.mean(auroc_HSIC), np.std(auroc_HSIC, ddof=1)),\
    "aupr:%.2f$\\pm$%.2f" %(np.mean(aupr_HSIC), np.std(aupr_HSIC, ddof=1)))