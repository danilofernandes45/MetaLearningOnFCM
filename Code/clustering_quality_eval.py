import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
from fuzzy_cmeans import *
from math import sqrt

def computeMAPE(y_test, y_predicted):
    n = y_test.size
    scaled_abs_dif = abs(y_test - y_predicted) / y_test
    return scaled_abs_dif.sum() / n

def computeRRMSE(y_test, y_predicted):
    n = y_test.size
    mean_y_test = y_test.mean()
    sum_squared_dif_test_pred = ( ( y_test - y_predicted )**2 ).sum()
    sum_squared_dif_test_mean = ( ( y_test - mean_y_test )**2 ).sum()
    return sqrt( sum_squared_dif_test_pred / sum_squared_dif_test_mean )

#Campello & Hruschka (2006)
def fuzzy_silhouette(X, u):
    D = pairwise_distances(X)
    c, n = u.shape
    s = np.zeros(n)

    for i in range(n):
        a = 1e-5
        b = 1e5
        for k in range(c):
            # Ponderação das distâncias por grau de pertencimento
            weights = u[k]  # todos os graus de pertinência para cluster k
            d = np.sum(weights * D[i]) / np.sum(weights)
            if k == np.argmax(u[:, i]):
                a = d
            else:
                b = min(b, d)
        s[i] = (b - a) / max(a, b)
    return np.mean(s)

def getSyntheticData():
    datasets = []
    n_centers = []

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
            datasets.append(X)
            n_centers.append(c)

    return datasets, n_centers

def getRealData():
    datasets = []
    n_centers = []

    np.random.seed(123)
    desc_datasets = pd.read_csv("../Data/Real/Real_Datasets_OpenML/desc_datasets.csv")
    size = desc_datasets.shape[0]

    for i in range(size):

        df = pd.read_csv("../Data/Real/Real_Datasets_OpenML/"+desc_datasets['Name'][i])
        p = desc_datasets['Num_Columns'][i]
        c = df.iloc[:,p].nunique() #Number of classes
        X = df.iloc[:,:p].to_numpy() #Datasets without targets
        X = minmax_scale(X)
        datasets.append(X)
        n_centers.append(c)
    
    return datasets, n_centers

def computeGroundTruth(estimate_m, data_func, output_filename):

    fsi = np.array([])
    val_list = ['B', 'FS', 'K', 'T']

    datasets, n_centers = data_func()
    N = len(datasets)

    for idx in range(N):
        fcm_func = lambda t : fuzzyCMeans(data=datasets[idx], c=n_centers[idx], seed = 123, m=estimate_m[t].iloc[idx])[0]
        fs_func = lambda t : fuzzy_silhouette(datasets[idx], fcm_func(t))
        metrics = list(map(fs_func, val_list))
        fsi = np.append( fsi, metrics )

    fsi = fsi.reshape([N, 4])
    np.savetxt(output_filename, fsi, delimiter = ",", fmt="%.5f")

def estimateBaseErrors(datasets, n_centers, estimate_m, estimate_fs):
    indices = ['B','FS','K','T']

    num_datasets = estimate_m.shape[0]

    mean_mape = np.zeros([30, 4])
    mean_rrmse = np.zeros([30, 4])

    median_mape = np.zeros([30, 4])
    median_rrmse = np.zeros([30, 4])

    for r in range(30):#Experiment
        print(f"Trial: {r}")
        idx = np.arange(0, num_datasets, 1)
        np.random.seed(r)
        np.random.shuffle(idx)

        for j in range(4):#Indices
            mean_mape_error = []
            mean_rrmse_error = []

            median_mape_error = []
            median_rrmse_error = []

            begin = 0
            end = num_datasets // 10

            for l in range(10):#10-Fold CrossValidation
                if(l < 9):
                    idx_train = np.append(idx[0:begin], idx[end:])
                    idx_test = idx[begin: end]
                else:
                    idx_train = idx[0:begin]
                    idx_test = idx[begin:]


                m_train = estimate_m[indices[j]].iloc[idx_train]
                m_mean = m_train.mean()
                m_median = m_train.median()

                xb_test = estimate_fs[indices[j]].iloc[idx_test]

                # fcm_func = lambda idx, m : fuzzyCMeans(data=datasets[idx], c=n_centers[idx], seed = 123, m=m)[:2]
                # xb_func = lambda idx, m : XieBeniIndex(datasets[idx], *fcm_func(idx, m), m)

                fcm_func = lambda idx, m : fuzzyCMeans(data=datasets[idx], c=n_centers[idx], seed = 123, m=m)[0]
                xb_func = lambda idx, m : fuzzy_silhouette(datasets[idx], fcm_func(idx, m))

                xb_mean_func = lambda idx : xb_func(idx, m_mean)
                xb_median_func = lambda idx : xb_func(idx, m_median)

                xb_mean = list(map(xb_mean_func, idx_test))
                xb_median = list(map(xb_median_func, idx_test))

                mean_mape_error.append(computeMAPE(xb_test, xb_mean))
                mean_rrmse_error.append(computeRRMSE(xb_test, xb_mean))

                median_mape_error.append(computeMAPE(xb_test, xb_median))
                median_rrmse_error.append(computeRRMSE(xb_test, xb_median))

                begin = end
                end += num_datasets // 10

            mean_mape[r, j] = np.array(mean_mape_error).mean()
            mean_rrmse[r, j] = np.array(mean_rrmse_error).mean()

            median_mape[r, j] = np.array(median_mape_error).mean()
            median_rrmse[r, j] = np.array(median_rrmse_error).mean()
    
    return mean_mape, mean_rrmse, median_mape, median_rrmse


def computeErrorBaselines(estimate_m, estimate_fs, data_func, path):

    datasets, n_centers = data_func()    

    mean_mape, mean_rrmse, median_mape, median_rrmse = estimateBaseErrors(datasets, n_centers, estimate_m, estimate_fs)    

    # np.savetxt("../Data/Synthetic/Clustering_Quality/Baselines/BruteForce/average_mape_error.csv", mean_mape, delimiter = ",", fmt="%.10f")
    np.savetxt(f"{path}/average_mape_error.csv", mean_mape, delimiter = ",", fmt="%.10f")
    np.savetxt(f"{path}/median_mape_error.csv", median_mape, delimiter = ",", fmt="%.10f")

    np.savetxt(f"{path}/average_rrmse_error.csv", mean_rrmse, delimiter = ",", fmt="%.10f")
    np.savetxt(f"{path}/median_rrmse_error.csv", median_rrmse, delimiter = ",", fmt="%.10f")


def computeErrorBaselinesReal():

    estimate_m = pd.read_csv("../Data/Real/Metadatasets/estimate_m_real.csv")
    estimate_xb = pd.read_csv("../Data/Real/Clustering_Quality/xb_real_ground_truth.csv")
    mtf_distances = pd.read_csv("../Data/Real/Metadatasets/mf_distances_real.csv", header = None)
    
    datasets, n_centers = getRealData()

    indices = ['B','FS','K','T']

    num_datasets = estimate_m.shape[0]

    mean_mape = np.zeros([30, 4])
    mean_rrmse = np.zeros([30, 4])

    median_mape = np.zeros([30, 4])
    median_rrmse = np.zeros([30, 4])

    for r in range(30):#Experiment
        print(f"Trial: {r}")
        idx = np.arange(0, num_datasets, 1)
        np.random.seed(r)
        np.random.shuffle(idx)

        for j in range(4):#Indices
            mean_mape_error = []
            mean_rrmse_error = []

            median_mape_error = []
            median_rrmse_error = []

            begin = 0
            end = num_datasets // 10

            for l in range(10):#10-Fold CrossValidation
                if(l < 9):
                    idx_train = np.append(idx[0:begin], idx[end:])
                    idx_test = idx[begin: end]
                else:
                    idx_train = idx[0:begin]
                    idx_test = idx[begin:]


                m_train = estimate_m[indices[j]].iloc[idx_train]
                m_mean = m_train.mean()
                m_median = m_train.median()

                xb_test = estimate_xb[indices[j]].iloc[idx_test]

                # fcm_func = lambda idx, m : fuzzyCMeans(data=datasets[idx], c=n_centers[idx], seed = 123, m=m)[:2]
                # xb_func = lambda idx, m : XieBeniIndex(datasets[idx], *fcm_func(idx, m), m)
                fcm_func = lambda idx, m : fuzzyCMeans(data=datasets[idx], c=n_centers[idx], seed = 123, m=m)[0]
                xb_func = lambda idx, m : fuzzy_silhouette(datasets[idx], fcm_func(idx, m))

                xb_mean_func = lambda idx : xb_func(idx, m_mean)
                xb_median_func = lambda idx : xb_func(idx, m_median)

                xb_mean = list(map(xb_mean_func, idx_test))
                xb_median = list(map(xb_median_func, idx_test))

                mean_mape_error.append(computeMAPE(xb_test, xb_mean))
                mean_rrmse_error.append(computeRRMSE(xb_test, xb_mean))

                median_mape_error.append(computeMAPE(xb_test, xb_median))
                median_rrmse_error.append(computeRRMSE(xb_test, xb_median))

                begin = end
                end += num_datasets // 10

            mean_mape[r, j] = np.array(mean_mape_error).mean()
            mean_rrmse[r, j] = np.array(mean_rrmse_error).mean()

            median_mape[r, j] = np.array(median_mape_error).mean()
            median_rrmse[r, j] = np.array(median_rrmse_error).mean()

    np.savetxt("../Data/Real/Clustering_Quality/Baselines/BruteForce/average_mape_error.csv", mean_mape, delimiter = ",", fmt="%.10f")
    np.savetxt("../Data/Real/Clustering_Quality/Baselines/BruteForce/median_mape_error.csv", median_mape, delimiter = ",", fmt="%.10f")

    np.savetxt("../Data/Real/Clustering_Quality/Baselines/BruteForce/average_rrmse_error.csv", mean_rrmse, delimiter = ",", fmt="%.10f")
    np.savetxt("../Data/Real/Clustering_Quality/Baselines/BruteForce/median_rrmse_error.csv", median_rrmse, delimiter = ",", fmt="%.10f")


#DATA
estimate_real_m = pd.read_csv("../Data/Real/Metadatasets/estimate_m_real.csv")
estimate_syn_m = pd.read_csv("../Data/Synthetic/Metadatasets/estimate_m_simulated.csv")

estimate_syn_fs = pd.read_csv("../Data/Synthetic/Clustering_Quality/fs_simulated_ground_truth.csv")


# computeGroundTruth(estimate_syn_m, getSyntheticData, "fs_simulated_ground_truth.csv")
path = "../Data/Synthetic/Clustering_Quality/Baselines/BruteForce"
computeErrorBaselines(estimate_syn_m, estimate_syn_fs, getSyntheticData, path)