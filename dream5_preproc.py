import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import Nystroem
feature_map_nystroem = Nystroem(gamma=1/8, random_state=10, n_components=10)

#coldstress_data = pd.read_csv('data/training data/Network 1 - in silico/net1_expression_data.tsv','\t')
coldstress_data = pd.read_csv('data/training data/Network2 - S.aureus/net2_expression_data.tsv','\t')
cols = coldstress_data.columns

SK = 16


cold = coldstress_data.values
from sklearn.cluster import KMeans
import numpy as np
cold.astype(np.float64)
kmeans = KMeans(n_clusters=SK, random_state=0).fit(cold)
coldstress_data = pd.DataFrame(kmeans.cluster_centers_)
coldstress_data.columns = cols

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
    
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

H = np.matlib.identity(SK, dtype = float) - 1.0/SK * np.ones((SK,SK))
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
  #kel=normalize(kel, norm = 'max')
  Min = min(Min,np.min(kel))
  Max = max(Max,np.max(kel))

print(Min,Max)

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
  #print(kel)
  kel = np.uint16(kel*65535)
  #print(kel)
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
#pr.to_csv('data/test data/DREAM5_NetworkInference_Network2_cluster_%d.tsv' %SK,'\t',index=False)
pr.to_csv('data/test data/DREAM5_NetworkInference_Network2_GK.tsv','\t',index=False)

