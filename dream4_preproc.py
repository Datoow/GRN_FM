import pandas as pd
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

num = 3
#num = int(input("num?"))
coldstress_data = pd.read_csv('data/test data/insilico_size100_%d_dream4_timeseries.tsv' %num,'\t')

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

Min = 100
Max = -100

for i in range(1, len(cols)):
  exp = np.array(coldstress_data[[cols[i]]].values.tolist())
  #if i == 91 or i == 21:
    #print(i)
    #print(exp.flatten().tolist(),sep=',')
  #print(exp)
  kel = kernel_gaussian(exp, exp, np.sqrt(len(exp)))
  #print(kel)
  kel = kel / (np.linalg.norm(kel, 'fro') + 10e-10)
  
  Ht = H.copy()
  
  Ht = H @ kel
  
  kel = Ht @ H
  
  kel = kel.flatten()
  
  kel = kel.reshape(1,-1)
  
  Min = min(Min, np.min(kel))
  Max = max(Max, np.max(kel))
  #print(kel)
  #kel=normalize(kel, norm = 'max')
print(Min, Max)

for i in range(1, len(cols)):
  exp = np.array(coldstress_data[[cols[i]]].values.tolist())
  #exp = exp [indexl]
  #exp[0] = exp[0]+0.000000001
  kel = kernel_gaussian(exp, exp, np.sqrt(len(exp)))
  #print(kel)
  kel = kel / (np.linalg.norm(kel, 'fro') + 10e-10)
  
  Ht = H.copy()
  
  Ht = H @ kel
  
  kel = Ht @ H
  
  kel = kel.flatten()
  
  kel = kel.reshape(1,-1)
  
  kel = (kel - Min) / (Max - Min)
  
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

pr = pd.DataFrame(E)
pr.columns = cols[1:]
#print(pr.columns)
pr.to_csv('data/test data/DREAM5_NetworkInference_Net%d_Gauss2.tsv' %num,'\t',index=False)

pr = pd.DataFrame(E2)
pr.columns = cols[1:]
#print(pr.columns)
pr.to_csv('data/test data/DREAM5_NetworkInference_Net%d_Exp.tsv' %num,'\t',index=False)

kernel_data2 = pd.DataFrame(E3)
kernel_data2.columns = cols[1:]
kernel_data2.to_csv('data/test data/DREAM5_NetworkInference_Net%d_Gauss3.tsv' %num,'\t',index=False)

#,header=None
#Gold_Net = pd.read_csv('data/test data/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv','\t',header=None)
#num = int(input("num?"))
Gold_Net = pd.read_csv('data/test data/insilico_size100_%d_goldstandard.tsv' %num,'\t',header=None)


Gold_Net = Gold_Net.values
#np.random.shuffle(Gold_Net)

GN_X = Gold_Net[:,0:2]
GN_Y = Gold_Net[:,2]

Half = np.sum(GN_Y == 1)
print("Half:", Half)

if len(GN_Y[GN_Y==2]) > 0:
  GN_X = GN_X[1:]
  GN_Y = GN_Y[1:]


GN_OX = GN_X
GN_OY = GN_Y
print(GN_OX.shape)
print(GN_OY.shape)

#index = [i for i in range(Half*2)]

#kernel_data = pd.read_csv('data/test data/DREAM5_NetworkInference_Network2_Gaussiankernel100.tsv','\t')
kernel_data = pd.read_csv('data/test data/DREAM5_NetworkInference_Net%d_Gauss2.tsv' %num,'\t')
exp_data = pd.read_csv('data/test data/DREAM5_NetworkInference_Net%d_Exp.tsv' %num,'\t')

avr_auroc = 0.0
avr_aupr = 0.0

#hsic = np.zeros((1,1))
#GN_OM = np.zeros((1,2*160*10))
f = open('data/test data/DREAM5_NetworkInference_Net%d_Gene_Ori_Map.tsv' %num, 'w')
f.truncate()
f.close()
f = open('data/test data/DREAM5_NetworkInference_Net%d_HSIC.tsv' %num, 'w')
f.truncate()
f.close()
f = open('data/test data/DREAM5_NetworkInference_Net%d_PC.tsv' %num, 'w')
f.truncate()
f.close()

for i in range(len(GN_OX)):
  k_data = np.array(kernel_data[GN_OX[i]].values.tolist())
  e_data = np.array(exp_data[GN_OX[i]].values.tolist())
  k_data2 = np.array(kernel_data2[GN_OX[i]].values.tolist())
  #if i == 0:
  #  print(k_data2)
  #GN_OM = np.vstack((GN_OM, np.hstack((exp_data[:,0].T,exp_data[:,1].T))))
  #hsic = np.vstack((hsic, sum(exp_data[:,0].T*exp_data[:,1].T)))
  #print(np.hstack((exp_data[:,0].T,exp_data[:,1].T)).shape)
  with open('data/test data/DREAM5_NetworkInference_Net%d_Gene_Ori_Map.tsv' %num, 'a') as csvfile:
    employee_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(np.hstack((k_data[:,0].T,k_data[:,1].T)))
  
  with open('data/test data/DREAM5_NetworkInference_Net%d_HSIC.tsv' %num, 'a') as csvfile:
    employee_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow([sum(k_data2[:,0].T*k_data2[:,1].T/1e9)])
    
  with open('data/test data/DREAM5_NetworkInference_Net%d_PC.tsv' %num, 'a') as csvfile:
    employee_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow([abs(pearsonr(e_data[:,0].T,e_data[:,1].T)[0])])
  
  
hsic = pd.read_csv('data/test data/DREAM5_NetworkInference_Net%d_HSIC.tsv' %num,'\t',header = None)
hsic = np.array(hsic.values.tolist()).flatten()
hsic = hsic.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler() 
hsic = min_max_scaler.fit_transform(hsic)
hsic = hsic.flatten()
HSIC = pd.DataFrame(hsic)
HSIC.to_csv('data/test data/DREAM5_NetworkInference_Net%d_HSIC2.tsv',index=False)

pc = pd.read_csv('data/test data/DREAM5_NetworkInference_Net%d_PC.tsv' %num,'\t',header = None)
Gold_Net = pd.read_csv('data/test data/insilico_size100_%d_goldstandard.tsv' %num,'\t',header=None)
Gold_Net = Gold_Net.values

GN_X = Gold_Net[:,0:2]
GN_Y = Gold_Net[:,2]

Half = np.sum(GN_Y == 1)
print("Half:", Half)


GN_OX = GN_X
GN_OY = GN_Y.astype(np.int)

print("PC")
avr_auroc = roc_auc_score(GN_OY,pc)
avr_aupr = average_precision_score(GN_OY,pc)
print("auroc:",avr_auroc)
print("aupr:",avr_aupr)

print("HSIC")
avr_auroc = roc_auc_score(GN_OY,hsic)
avr_aupr = average_precision_score(GN_OY,hsic)
print("auroc:",avr_auroc)
print("aupr:",avr_aupr)

