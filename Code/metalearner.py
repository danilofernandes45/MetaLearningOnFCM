from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from math import sqrt

NUM_META_FEATURES = 3
NUM_VALIDATION_INDICES = 10
NUM_EXPERIMENTS = 30
NUM_ALGORITHMS = 3
NUM_FOLDS = 10

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

estimate_m = pd.read_csv("../Data/Real/Metadatasets/estimate_m_real.csv")
mtf_density = pd.read_csv("../Data/Real/Metadatasets/mf_density_real.csv", header = None)
mtf_distances = pd.read_csv("../Data/Real/Metadatasets/mf_distances_real.csv", header = None)
mtf_dist_cor = pd.read_csv("../Data/Real/Metadatasets/mf_dist_corr_real.csv", header = None)

metafeatures = [mtf_density, mtf_distances, mtf_dist_cor]

knn_params = np.arange(1, 11) #Number of neighbors

svm_rbf_params = np.meshgrid([0.1, 1, 10, 100, 1e3, 1e4], [0.1, 1, 10, 100, 1e3, 1e4]) #(C, gamma)
svm_rbf_params[0] = svm_rbf_params[0].flatten()
svm_rbf_params[1] = svm_rbf_params[1].flatten()
svm_rbf_params = np.array(svm_rbf_params).T

random_forest_params = np.array([10, 20, 40, 60, 80, 100, 200, 300, 400, 500]) #Number of decision trees

alg_params = [knn_params, svm_rbf_params, random_forest_params]

algorithms = ["knn", "svm_rbf", "random_forest"]

num_datasets = estimate_m.shape[0]
for i in range(NUM_META_FEATURES):#Metafeatures
    for j in range(NUM_VALIDATION_INDICES):#Validation indeces
        for alg in range(NUM_ALGORITHMS):#Algorithms

            if(algorithms[alg] != "random_forest"):
                continue

            num_combinations_params = alg_params[alg].shape[0]
            mape = np.zeros([NUM_EXPERIMENTS, num_combinations_params])
            mrae = np.zeros([NUM_EXPERIMENTS, num_combinations_params])
            rrmse = np.zeros([NUM_EXPERIMENTS, num_combinations_params])

            for p in range(num_combinations_params):#Algorithm's Parameters
                print(alg_params[alg][p])
                model = None
                if(algorithms[alg] == "knn"):
                    model = KNeighborsRegressor(n_neighbors = alg_params[alg][p], weights = 'distance')
                elif(algorithms[alg] == "svm_rbf"):
                    model = SVR(kernel = "rbf", C = alg_params[alg][p][0], gamma = alg_params[alg][p][1])
                elif(algorithms[alg] == "random_forest"):
                    model = RandomForestRegressor(n_estimators = alg_params[alg][p], min_samples_split = 0.02, min_impurity_decrease = 0.1, random_state = 123)

                for r in range(NUM_EXPERIMENTS):#Experiment
                    idx = np.arange(0, num_datasets, 1)
                    np.random.seed(r)
                    np.random.shuffle(idx)
                    begin = 0
                    end = num_datasets // NUM_FOLDS
                    mape_error = []
                    mrae_error = []
                    rrmse_error = []
                    for l in range(NUM_FOLDS):#10-Fold CrossValidation
                        if( l < (NUM_FOLDS - 1) ):
                            idx_train = np.append(idx[0:begin], idx[end:])
                            idx_test = idx[begin: end]
                        else:
                            idx_train = idx[0:begin]
                            idx_test = idx[begin:]

                        X_train = metafeatures[i].iloc[idx_train,:]
                        X_test = metafeatures[i].iloc[idx_test,:]
                        y_train = estimate_m[indices[j]].iloc[idx_train]
                        y_test = estimate_m[indices[j]].iloc[idx_test]

                        model.fit(X_train, y_train)
                        y_predicted = model.predict(X_test)
                        mape_error.append(computeMAPE(y_test, y_predicted))
                        mrae_error.append(computeMRAE(y_test, y_predicted))
                        rrmse_error.append(computeRRMSE(y_test, y_predicted))

                        begin = end
                        end += num_datasets // NUM_FOLDS

                    mape[r, p] = np.array(mape_error).mean()
                    mrae[r, p] = np.array(mrae_error).mean()
                    rrmse[r, p] = np.array(rrmse_error).mean()

            np.savetxt("../Data/Real/Error_Datasets/"+algorithms[alg]+"/"+metafeatures_names[i]+"/"+indices_names[j]+"_mape.csv", mape, delimiter = ",", fmt="%.10f")
            np.savetxt("../Data/Real/Error_Datasets/"+algorithms[alg]+"/"+metafeatures_names[i]+"/"+indices_names[j]+"_mrae.csv", mrae, delimiter = ",", fmt="%.10f")
            np.savetxt("../Data/Real/Error_Datasets/"+algorithms[alg]+"/"+metafeatures_names[i]+"/"+indices_names[j]+"_rrmse.csv", rrmse, delimiter = ",", fmt="%.10f")
