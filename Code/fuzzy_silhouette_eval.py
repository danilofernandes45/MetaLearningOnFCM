import numpy as np
from skfuzzy.cluster import cmeans
from sklearn.metrics import pairwise_distances
import pandas as pd
from fuzzy_cmeans import *

def fuzzy_silhouette_score(data, u, cntr):
    print(data.shape)
    print(u.shape)
    print(cntr.shape)
    N = data.shape[0]  # Número de pontos
    c = cntr.shape[0]  # Número de clusters
    
    # Calcula distâncias entre os pontos e os centros dos clusters
    dist_to_centers = pairwise_distances(data, cntr, metric='euclidean')
    
    # Inicializa arrays de a_i e b_i
    a_i = np.zeros(N)
    b_i = np.full(N, np.inf)

    for i in range(N):
        # Calcula a_i (distância intra-cluster ponderada)
        for j in range(c):
            a_i[i] += u[j, i] * dist_to_centers[i, j]
        
        # Calcula b_i (distância ao cluster mais próximo)
        for j in range(c):
            for k in range(c):
                if j != k:
                    b_ik = u[k, i] * dist_to_centers[i, k]
                    b_i[i] = min(b_i[i], b_ik)

    # Calcula Silhueta para cada ponto e cluster
    S_ij = (b_i - a_i) / np.maximum(a_i, b_i)

    # Calcula o Fuzzy Silhouette Index (FSI)
    FSI = np.sum(u * S_ij[:, np.newaxis]) / N
    return FSI

def computeSynthetic():

    estimate_m = pd.read_csv("../Data/Synthetic/Metadatasets/estimate_m_simulated.csv")
    mtf_distances = pd.read_csv("../Data/Synthetic/Metadatasets/mf_distances_simulated.csv", header = None)
    fsi = np.array([])

    np.random.seed(123)
    n_samples = np.random.choice([200, 300, 400, 500, 600], 234)
    id = 0 # min = 0
    for p in range(5, 31): #min 5, max = 31
        np.random.seed(123)
        centers = 10 * np.random.random(10*p).reshape([10,p])
        np.random.seed(123)
        std = 5 * np.random.random(10*p).reshape([10,p])
        for c in range(2, 11):
            X, _ = make_blobs(n_samples = n_samples[id], n_features = p, cluster_std = std[0:c], centers = centers[0:c], shuffle=False, random_state = 123)
            X = minmax_scale(X)

            fcm_func = lambda m : fuzzyCMeans(data=X, c=c, seed = 123, m=m)[:2]
            # fsi_func = lambda m : fuzzy_silhouette_score(X, *fcm_func(m))
            fsi_func = lambda m : XieBeniIndex(X, *fcm_func(m), m)

            row = estimate_m.iloc[id]
            row = row[['B', 'FS', 'K', 'T']]
            metrics = list(map(fsi_func, row))

            print("%d - %d - %d"%(id, p, c))
            fsi = np.append( fsi, metrics )

            id += 1

    fsi = fsi.reshape([234, 4])
    np.savetxt("fsi_simulated.csv", fsi, delimiter = ",", fmt="%.5f")


computeSynthetic()
