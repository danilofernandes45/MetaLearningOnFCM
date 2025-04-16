import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
from fuzzy_cmeans import *
from math import sqrt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

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
            sum_weights = np.sum(weights)
            d = 0
            if sum_weights > 0:
                d = np.sum(weights * D[i]) / sum_weights
            if k == np.argmax(u[:, i]):
                a = d
            else:
                b = min(b, d)
        s[i] = (b - a) / max(max(a, b), 1e-10)
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

def evaluateBase(data_func, estimate_m, path):
    indices = ['B','FS','K','T']
    datasets, n_centers = data_func()
    num_datasets = estimate_m.shape[0]
    mean_mape = np.zeros([30, 4])
    median_mape = np.zeros([30, 4])

    for r in range(30):#Experiment
        print(f"Trial: {r}")
        idx = np.arange(0, num_datasets, 1)
        np.random.seed(r)
        np.random.shuffle(idx)

        for j in range(4):#Indices
            mean_base = []
            median_base = []

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

                # fcm_func = lambda idx, m : fuzzyCMeans(data=datasets[idx], c=n_centers[idx], seed = 123, m=m)[:2]
                # xb_func = lambda idx, m : XieBeniIndex(datasets[idx], *fcm_func(idx, m), m)

                fcm_func = lambda idx, m : fuzzyCMeans(data=datasets[idx], c=n_centers[idx], seed = 123, m=m)[0]
                fs_func = lambda idx, m : fuzzy_silhouette(datasets[idx], fcm_func(idx, m))

                fs_mean_func = lambda idx : fs_func(idx, m_mean)
                fs_median_func = lambda idx : fs_func(idx, m_median)

                fs_mean = np.mean(list(map(fs_mean_func, idx_test)))
                fs_median = np.mean(list(map(fs_median_func, idx_test)))

                mean_base.append(fs_mean)
                median_base.append(fs_median)

                begin = end
                end += num_datasets // 10

            mean_mape[r, j] = np.mean(mean_base)
            median_mape[r, j] = np.mean(median_base)
    
    np.savetxt(f"{path}/base_mean_fuzzy_silhouette.csv", mean_mape, delimiter = ",", fmt="%.10f")
    np.savetxt(f"{path}/base_median_fuzzy_silhouette.csv", median_mape, delimiter = ",", fmt="%.10f")

def estimateErrors(data_func, estimate_m, estimate_fs, models, metadatasets, path):
    indices = ['B','FS','K','T']
    datasets, n_centers = data_func()
    num_datasets = estimate_m.shape[0]
    mape = np.zeros([30, 4])
    rrmse = np.zeros([30, 4])

    for r in range(30):#Experiment
        print(f"Trial: {r}")
        idx = np.arange(0, num_datasets, 1)
        np.random.seed(r)
        np.random.shuffle(idx)

        for j in range(4):
            val_id = indices[j]
            mape_error = []
            rrmse_error = []

            begin = 0
            end = num_datasets // 10

            for l in range(10):#10-Fold CrossValidation
                if(l < 9):
                    idx_train = np.append(idx[0:begin], idx[end:])
                    idx_test = idx[begin: end]
                else:
                    idx_train = idx[0:begin]
                    idx_test = idx[begin:]

                test_size = len(idx_test)

                X_train = metadatasets[val_id].iloc[idx_train]
                X_test = metadatasets[val_id].iloc[idx_test]
                m_train = estimate_m[val_id].iloc[idx_train]                
                fs_test = estimate_fs[val_id].iloc[idx_test]

                models['mape'][val_id].fit(X_train, m_train)
                models['rrmse'][val_id].fit(X_train, m_train)
                m_pred_mape = models['mape'][val_id].predict(X_test)
                m_pred_rrmse = models['rrmse'][val_id].predict(X_test)

                fcm_func = lambda i, m : fuzzyCMeans(data=datasets[idx_test[i]], c=n_centers[idx_test[i]], seed = 123, m=m)[0]
                fs_func = lambda i, m : fuzzy_silhouette(datasets[idx_test[i]], fcm_func(i, m))

                fs_mape = lambda i : fs_func(i, m_pred_mape[i])
                fs_rrmse = lambda i : fs_func(i, m_pred_rrmse[i])

                fs_pred_mape = list(map(fs_mape, range(test_size)))
                fs_pred_rrmse = list(map(fs_rrmse, range(test_size)))

                mape_error.append(computeMAPE(fs_test, fs_pred_mape))
                rrmse_error.append(computeRRMSE(fs_test, fs_pred_rrmse))

                begin = end
                end += num_datasets // 10

            mape[r, j] = np.array(mape_error).mean()
            rrmse[r, j] = np.array(rrmse_error).mean()
    
    np.savetxt(f"{path}/mape_error.csv", mape, delimiter = ",", fmt="%.10f")
    np.savetxt(f"{path}/rrmse_error.csv", rrmse, delimiter = ",", fmt="%.10f")
 

# def computeErrorBaselines(estimate_m, estimate_fs, data_func, path):

#     datasets, n_centers = data_func()    

#     mean_mape, mean_rrmse, median_mape, median_rrmse = estimateBaseErrors(datasets, n_centers, estimate_m, estimate_fs)    

#     # np.savetxt("../Data/Synthetic/Clustering_Quality/Baselines/BruteForce/average_mape_error.csv", mean_mape, delimiter = ",", fmt="%.10f")
#     np.savetxt(f"{path}/average_mape_error.csv", mean_mape, delimiter = ",", fmt="%.10f")
#     np.savetxt(f"{path}/median_mape_error.csv", median_mape, delimiter = ",", fmt="%.10f")

#     np.savetxt(f"{path}/average_rrmse_error.csv", mean_rrmse, delimiter = ",", fmt="%.10f")
#     np.savetxt(f"{path}/median_rrmse_error.csv", median_rrmse, delimiter = ",", fmt="%.10f")

def getRealParamsMtL():
    data_func = getRealData
    estimate_m = pd.read_csv("../Data/Real/Metadatasets/estimate_m_real.csv")
    estimate_fs = pd.read_csv("../Data/Real/Clustering_Quality/fs_real_ground_truth.csv")
    metadatasets = pd.read_csv("../Data/Real/Metadatasets/mf_distances_real.csv", header=None)
    path = "../Data/Real/Clustering_Quality/MtL"

    models = {'mape': {}, 'rrmse': {}}
    models['mape']['B'] = KNeighborsRegressor(n_neighbors = 10, weights = 'distance')
    models['mape']['FS'] = SVR(kernel = "rbf", C = 0.1, gamma = 10)
    models['mape']['K'] = KNeighborsRegressor(n_neighbors = 8, weights = 'distance')
    models['mape']['T'] = KNeighborsRegressor(n_neighbors = 9, weights = 'distance')

    models['rrmse']['B'] = KNeighborsRegressor(n_neighbors = 8, weights = 'distance')
    models['rrmse']['FS'] = SVR(kernel = "rbf", C = 1, gamma = 100)
    models['rrmse']['K'] = SVR(kernel = "rbf", C = 10, gamma = 0.1)
    models['rrmse']['T'] = SVR(kernel = "rbf", C = 10, gamma = 0.1)

    return data_func, estimate_m, estimate_fs, models, metadatasets, path

def getRealParamsMtL():
    data_func = getRealData
    estimate_m = pd.read_csv("../Data/Real/Metadatasets/estimate_m_real.csv")
    estimate_fs = pd.read_csv("../Data/Real/Clustering_Quality/fs_real_ground_truth.csv")
    metadatasets = pd.read_csv("../Data/Real/Metadatasets/mf_distances_real.csv", header=None)
    path = "../Data/Real/Clustering_Quality/MtL"

    models = {'mape': {}, 'rrmse': {}}
    models['mape']['B'] = KNeighborsRegressor(n_neighbors = 10, weights = 'distance')
    models['mape']['FS'] = SVR(kernel = "rbf", C = 0.1, gamma = 10)
    models['mape']['K'] = KNeighborsRegressor(n_neighbors = 8, weights = 'distance')
    models['mape']['T'] = KNeighborsRegressor(n_neighbors = 9, weights = 'distance')

    models['rrmse']['B'] = KNeighborsRegressor(n_neighbors = 8, weights = 'distance')
    models['rrmse']['FS'] = SVR(kernel = "rbf", C = 1, gamma = 100)
    models['rrmse']['K'] = SVR(kernel = "rbf", C = 10, gamma = 0.1)
    models['rrmse']['T'] = SVR(kernel = "rbf", C = 10, gamma = 0.1)

    return data_func, estimate_m, estimate_fs, models, metadatasets, path

def getSyntheticParamsMtL():
    data_func = getSyntheticData
    estimate_m = pd.read_csv("../Data/Synthetic/Metadatasets/estimate_m_simulated.csv")
    estimate_fs = pd.read_csv("../Data/Synthetic/Clustering_Quality/fs_simulated_ground_truth.csv")
    metadatasets = {}
    metadatasets['B'] = pd.read_csv("../Data/Synthetic/Metadatasets/mf_dist_corr_simulated.csv", header=None)
    metadatasets['FS'] = pd.read_csv("../Data/Synthetic/Metadatasets/mf_distances_simulated.csv", header=None)
    metadatasets['K'] = metadatasets['FS']
    metadatasets['T'] = metadatasets['B']
    path = "../Data/Synthetic/Clustering_Quality/MtL"

    models = {'mape': {}, 'rrmse': {}}
    models['mape']['B'] = KNeighborsRegressor(n_neighbors = 10, weights = 'distance')
    models['mape']['FS'] = SVR(kernel = "rbf", C = 0.1, gamma = 1)
    models['mape']['K'] = SVR(kernel = "rbf", C = 0.1, gamma = 10)
    models['mape']['T'] = KNeighborsRegressor(n_neighbors = 9, weights = 'distance')

    models['rrmse']['B'] = KNeighborsRegressor(n_neighbors = 10, weights = 'distance')
    models['rrmse']['FS'] = SVR(kernel = "rbf", C = 1, gamma = 10)
    models['rrmse']['K'] = SVR(kernel = "rbf", C = 10, gamma = 1)
    models['rrmse']['T'] = SVR(kernel = "rbf", C = 10, gamma = 0.1)

    return data_func, estimate_m, estimate_fs, models, metadatasets, path

def getSyntheticParamsBase():
    data_func = getSyntheticData
    estimate_m = pd.read_csv("../Data/Synthetic/Metadatasets/estimate_m_simulated.csv")
    path = "../Data/Synthetic/Clustering_Quality/Baselines/BruteForce"
    return data_func, estimate_m, path

def getRealParamsBase():
    data_func = getRealData
    estimate_m = pd.read_csv("../Data/Real/Metadatasets/estimate_m_real.csv")
    path = "../Data/Real/Clustering_Quality/Baselines/BruteForce"
    return data_func, estimate_m, path

#DATA
# estimate_real_m = pd.read_csv("../Data/Real/Metadatasets/estimate_m_real.csv")
# estimate_syn_m = pd.read_csv("../Data/Synthetic/Metadatasets/estimate_m_simulated.csv")

# estimate_syn_fs = pd.read_csv("../Data/Synthetic/Clustering_Quality/fs_simulated_ground_truth.csv")


# # computeGroundTruth(estimate_syn_m, getSyntheticData, "fs_simulated_ground_truth.csv")
# path = "../Data/Synthetic/Clustering_Quality/Baselines/BruteForce"
# computeErrorBaselines(estimate_syn_m, estimate_syn_fs, getSyntheticData, path)

evaluateBase(*getSyntheticParamsBase())

evaluateBase(*getRealParamsBase())

