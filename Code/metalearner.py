from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from math import sqrt

def computeMAPE(y_test, y_predicted):
    n = y_test.size
    scaled_abs_dif = abs(y_test - y_predicted) / y_test
    return scaled_abs_dif.sum() / n

def computeMRAE(y_test, y_predicted):
    n = y_test.size
    mean_y_test = y_test.mean()
    abs_dif_test_pred = abs(y_test - y_predicted)
    abs_dif_test_mean = abs(y_test - mean_y_test)
    return ( abs_dif_test_pred / abs_dif_test_mean ).sum() / n

def computeRRMSE(y_test, y_predicted):
    n = y_test.size
    mean_y_test = y_test.mean()
    sum_squared_dif_test_pred = ( ( y_test - y_predicted )**2 ).sum()
    sum_squared_dif_test_mean = ( ( y_test - mean_y_test )**2 ).sum()
    return sqrt( sum_squared_dif_test_pred / sum_squared_dif_test_mean )

indices_names = ["Z_Index", "PCAES_Index", "SCG_Index", "FS_Index", "XB_Index", "B_Index", "K_Index", "T_Index", "FH_Index", "AWCD_Index"]
metafeatures_names = ["Metafeature_Density", "Metafeature_Distance", "Metafeature_Dist_Cor"]

indices = ['Z','PCAES','SCG','FS','XB','B','K','T','FH','AWCD']

estimate_m = pd.read_csv("Metadatasets/estimate_m_simulated.csv")
mtf_density = pd.read_csv("Metadatasets/mf_density_simulated.csv", header = None)
mtf_distances = pd.read_csv("Metadatasets/mf_distances_simulated.csv", header = None)
mtf_dist_cor = pd.read_csv("Metadatasets/mf_dist_corr_simulated.csv", header = None)

metafeatures = [mtf_density, mtf_distances, mtf_dist_cor]

num_datasets = estimate_m.shape[0]
for i in range(3):#Metafeatures
    for j in range(10):#Validation indeces
        #Errors Experiment X Number of neighbors
        mape = np.zeros([30, 10])
        mrae = np.zeros([30, 10])
        rrmse = np.zeros([30, 10])
        for k in range(1, 11):#Number of neighbors
            knn = KNeighborsRegressor(n_neighbors = k, weights = 'distance')
            for r in range(30):#Experiment
                idx = np.arange(0, num_datasets, 1)
                np.random.seed(r)
                np.random.shuffle(idx)
                begin = 0
                end = num_datasets // 10
                mape_error = []
                mrae_error = []
                rrmse_error = []
                for l in range(10):#10-Fold CrossValidation
                    if(l < 9):
                        idx_train = np.append(idx[0:begin], idx[end:])
                        idx_test = idx[begin: end]
                    else:
                        idx_train = idx[0:begin]
                        idx_test = idx[begin:]

                    X_train = metafeatures[i].iloc[idx_train,:]
                    X_test = metafeatures[i].iloc[idx_test,:]
                    y_train = estimate_m[indices[j]].iloc[idx_train]
                    y_test = estimate_m[indices[j]].iloc[idx_test]

                    knn.fit(X_train, y_train)
                    y_predicted = knn.predict(X_test)
                    mape_error.append(computeMAPE(y_test, y_predicted))
                    mrae_error.append(computeMRAE(y_test, y_predicted))
                    rrmse_error.append(computeRRMSE(y_test, y_predicted))

                    begin = end
                    end += num_datasets // 10

                mape[r, (k-1)] = np.array(mape_error).mean()
                mrae[r, (k-1)] = np.array(mrae_error).mean()
                rrmse[r, (k-1)] = np.array(rrmse_error).mean()

        np.savetxt("Error_Datasets/"+metafeatures_names[i]+"/"+indices_names[j]+"/mape.csv", mape, delimiter = ",", fmt="%.10f")
        np.savetxt("Error_Datasets/"+metafeatures_names[i]+"/"+indices_names[j]+"/mrae.csv", mrae, delimiter = ",", fmt="%.10f")
        np.savetxt("Error_Datasets/"+metafeatures_names[i]+"/"+indices_names[j]+"/rrmse.csv", rrmse, delimiter = ",", fmt="%.10f")
