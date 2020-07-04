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
indices = ['Z','PCAES','SCG','FS','XB','B','K','T','FH','AWCD']

estimate_m = pd.read_csv("../Data/Real/Metadatasets/estimate_m_real.csv")

num_datasets = estimate_m.shape[0]

mean_mape = np.zeros([30, 10])
mean_mrae = np.zeros([30, 10])
mean_rrmse = np.zeros([30, 10])

mode_mape = np.zeros([30, 10])
mode_mrae = np.zeros([30, 10])
mode_rrmse = np.zeros([30, 10])

median_mape = np.zeros([30, 10])
median_mrae = np.zeros([30, 10])
median_rrmse = np.zeros([30, 10])

for r in range(30):#Experiment
    idx = np.arange(0, num_datasets, 1)
    np.random.seed(r)
    np.random.shuffle(idx)

    for j in range(10):#Indices
        mean_mape_error = []
        mean_mrae_error = []
        mean_rrmse_error = []

        mode_mape_error = []
        mode_mrae_error = []
        mode_rrmse_error = []

        median_mape_error = []
        median_mrae_error = []
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


            y_train = estimate_m[indices[j]].iloc[idx_train]
            y_test  = estimate_m[indices[j]].iloc[idx_test]

            mean_mape_error.append(computeMAPE(y_test, y_train.mean()))
            mean_mrae_error.append(computeMRAE(y_test, y_train.mean()))
            mean_rrmse_error.append(computeRRMSE(y_test, y_train.mean()))

            mode_mape_error.append(computeMAPE(y_test, y_train.mode()))
            mode_mrae_error.append(computeMRAE(y_test, y_train.mode()))
            mode_rrmse_error.append(computeRRMSE(y_test, y_train.mode()))

            median_mape_error.append(computeMAPE(y_test, y_train.median()))
            median_mrae_error.append(computeMRAE(y_test, y_train.median()))
            median_rrmse_error.append(computeRRMSE(y_test, y_train.median()))

            begin = end
            end += num_datasets // 10

        mean_mape[r, j] = np.array(mean_mape_error).mean()
        mean_mrae[r, j] = np.array(mean_mrae_error).mean()
        mean_rrmse[r, j] = np.array(mean_rrmse_error).mean()

        mode_mape[r, j] = np.array(mode_mape_error).mean()
        mode_mrae[r, j] = np.array(mode_mrae_error).mean()
        mode_rrmse[r, j] = np.array(mode_rrmse_error).mean()

        median_mape[r, j] = np.array(median_mape_error).mean()
        median_mrae[r, j] = np.array(median_mrae_error).mean()
        median_rrmse[r, j] = np.array(median_rrmse_error).mean()


np.savetxt("../Data/Real/Error_Datasets/average_mape_error.csv", mean_mape, delimiter = ",", fmt="%.10f")
# np.savetxt("../Data/Error_Datasets/marjority_mape_error.csv", mode_mape, delimiter = ",", fmt="%.10f")
np.savetxt("../Data/Real/Error_Datasets/median_mape_error.csv", median_mape, delimiter = ",", fmt="%.10f")

np.savetxt("../Data/Real/Error_Datasets/average_mrae_error.csv", mean_mrae, delimiter = ",", fmt="%.10f")
# np.savetxt("../Data/Error_Datasets/marjority_mrae_error.csv", mode_mrae, delimiter = ",", fmt="%.10f")
np.savetxt("../Data/Real/Error_Datasets/median_mrae_error.csv", median_mrae, delimiter = ",", fmt="%.10f")

np.savetxt("../Data/Real/Error_Datasets/average_rrmse_error.csv", mean_rrmse, delimiter = ",", fmt="%.10f")
# np.savetxt("../Data/Error_Datasets/marjority_rrmse_error.csv", mode_rrmse, delimiter = ",", fmt="%.10f")
np.savetxt("../Data/Real/Error_Datasets/median_rrmse_error.csv", median_rrmse, delimiter = ",", fmt="%.10f")
