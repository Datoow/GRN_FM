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


import random

store = pd.HDFStore('dendritic/dendritic_cell.h5')
#print(store.keys())
coldstress_data = store['/RPKMs']
store.close()
cols=coldstress_data.columns


#geneIDs=np.asarray(geneIDs,dtype=str)
#coldstress_data = coldstress_data.drop([0], axis=1)
#coldstress_data = coldstress_data.sample(n=300,random_state=123,axis=0)

#cols = coldstress_data.columns
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

SK = 128

H = np.matlib.identity(SK, dtype = float) - 1.0/SK * np.ones((SK,SK))

#indexl = random.sample(list(range(4126)), 128)

cold = coldstress_data.values
from sklearn.cluster import KMeans
import numpy as np
cold.astype(np.float64)
kmeans = KMeans(n_clusters=SK, random_state=0).fit(cold)
coldstress_data = pd.DataFrame(kmeans.cluster_centers_)
coldstress_data.columns = cols

Min = 100
Max = -100

for i in range(0, len(cols)):
  exp = np.array(coldstress_data[[cols[i]]].values.tolist())
  #exp = exp [indexl]
  exp[0] = exp[0]+0.000000001
  kel = kernel_gaussian(exp, exp, np.sqrt(len(exp)))
  #print(kel)
  kel = kel / (np.linalg.norm(kel, 'fro') + 10e-10)
  
  Ht = H.copy()
  
  Ht = H @ kel
  
  kel = Ht @ H
  
  kel = kel.flatten()
  
  kel = kel.reshape(1,-1)
  #print(kel)
  Min = min(Min, np.min(kel))
  Max = max(Max, np.max(kel))
  
print(Min, Max)

for i in range(0, len(cols)):
  exp = np.array(coldstress_data[[cols[i]]].values.tolist())
  #exp = exp [indexl]
  exp[0] = exp[0]+0.000000001
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
  if i == 0:
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
pr.columns = cols
#print(pr.columns)
pr.to_csv('data/test data/dendritic_Gauss2_%d.tsv' %SK,'\t',index=False)

pr = pd.DataFrame(E2)
pr.columns = cols
#print(pr.columns)
pr.to_csv('data/test data/dendritic_Exp_%d.tsv' %SK,'\t',index=False)

kernel_data2 = pd.DataFrame(E3)
kernel_data2.columns = cols
kernel_data2.to_csv('data/test data/dendritic_Gauss3_%d.tsv' %SK,'\t',index=False)

#,header=None
#Gold_Net = pd.read_csv('data/test data/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv','\t',header=None)
#num = int(input("num?"))

import re
h = {}
h2 = {}
s = open('dendritic/sc_gene_list.txt', 'r')  # gene symbol ID list of sc RNA-seq
for line in s:
    search_result = re.search(r'^([^\s]+)\s+([^\s]+)', line)
    #print("-",search_result.group(1),"-",search_result.group(2),"-")
    h[str(search_result.group(1))] = str(search_result.group(2))  # h [gene symbol] = gene ID
    h2[str(search_result.group(2))] = str(search_result.group(1)) #h2 geneID = gene symbol
geneID_map = h
ID_to_name_map = h2
s.close()

GN_X = []
GN_Y = []

unique_keys={}
s = open("dendritic/gold_standard_dendritic_whole.txt")  # 'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
for line in s:
    separation = line.split()
    #geneA_name, geneB_name, label = separation[0], separation[1], separation[2]
    #geneA_name = geneA_name.lower()
    #geneB_name = geneB_name.lower()
    separation[0] = h[separation[0]]
    separation[1] = h[separation[1]]
    GN_X.append(separation[:2])
    GN_Y.append(separation[2])
    
    
GN_X = np.array(GN_X)
GN_Y = np.array(GN_Y)


GN_OX = GN_X
GN_OY = GN_Y
print(GN_OX.shape)
print(GN_OY.shape)

#index = [i for i in range(Half*2)]

#kernel_data = pd.read_csv('data/test data/DREAM5_NetworkInference_Network2_Gaussiankernel100.tsv','\t')
kernel_data2 = pd.read_csv('data/test data/dendritic_Gauss3_%d.tsv' %SK,'\t')
kernel_data = pd.read_csv('data/test data/dendritic_Gauss2_%d.tsv' %SK,'\t')
exp_data = pd.read_csv('data/test data/dendritic_Exp_%d.tsv' %SK,'\t')
#print(kernel_data.columns)
#coldstress_data = coldstress_data.sample(n=10,random_state=123,axis=0)

#feature_map_nystroem = Nystroem(gamma=.2, random_state=1, n_components=50)
#print(len(coldstress_data[GN_X[1]]))
avr_auroc = 0.0
avr_aupr = 0.0

#hsic = np.zeros((1,1))
#GN_OM = np.zeros((1,2*160*10))
f = open('data/test data/dendritic_Gene_Ori_Map_%d.tsv' %SK, 'w')
f.truncate()
f.close()
f = open('data/test data/dendritic_HSIC_%d.tsv' %SK, 'w')
f.truncate()
f.close()
f = open('data/test data/dendritic_PC_%d.tsv' %SK, 'w')
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
  with open('data/test data/dendritic_Gene_Ori_Map_%d.tsv' %SK, 'a') as csvfile:
    employee_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(np.hstack((k_data[:,0].T,k_data[:,1].T)))
  
  with open('data/test data/dendritic_HSIC_%d.tsv' %SK, 'a') as csvfile:
    employee_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow([sum(k_data2[:,0].T*k_data2[:,1].T/1e9)])
    
  with open('data/test data/dendritic_PC_%d.tsv' %SK, 'a') as csvfile:
    employee_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow([abs(pearsonr(e_data[:,0].T,e_data[:,1].T)[0])])
  

'''
GN_Y = []
s = open("dendritic/gold_standard_dendritic_whole.txt")  # 'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
for line in s:
    separation = line.split()
    GN_Y.append(separation[2])
    

GN_Y = np.array(GN_Y)
GN_OY = GN_Y
print(GN_OY.shape)
'''


hsic = pd.read_csv('data/test data/dendritic_HSIC_%d.tsv' %SK,'\t',header = None)
hsic = np.array(hsic.values.tolist()).flatten()
hsic = hsic.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler() 
hsic = min_max_scaler.fit_transform(hsic)
hsic = hsic.flatten()
HSIC = pd.DataFrame(hsic)
HSIC.to_csv('data/test data/dendritic_HSIC2_%d.tsv' %SK,index=False)

pc = pd.read_csv('data/test data/dendritic_PC_%d.tsv' %SK,'\t',header = None)

GN_OY = '1' == GN_OY

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

