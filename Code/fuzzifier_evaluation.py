import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns

from estimation_fuzzifier import *

def trials_FCM(data, c, m, num_trials):
    best_membership_matrix = None
    best_centers = None
    best_obj_func = np.inf
    best_num_iter = None

    for i in range(num_trials):
        membership_matrix, centers, obj_func, num_iter = fuzzyCMeans(data, c, m, seed = i)
        if (obj_func < best_obj_func):
            best_membership_matrix = membership_matrix
            best_centers = centers
            best_obj_func = obj_func
            best_num_iter = num_iter

    return best_membership_matrix, best_centers, best_obj_func, best_num_iter

def evaluateFuzzifier(data, c, index):

    max_m = min(15, maxTheoreticalFuzzifier(data))
    #print(max_m)

    m_values = np.arange(1.02, max_m, 0.02)
    metrics = np.zeros([2, m_values.size])
    metrics[0] = m_values
    #print(m_values.size)

    for k in range(m_values.size):
        #print(k)
        membership_matrix, centers, obj_func, num_iter = trials_FCM(data, c, metrics[0, k], num_trials = 10)

        if (index == "Fukuyama-Sugeno"):
          metrics[1,k] = FukuyamaSugenoIndex(data, membership_matrix, centers, metrics[0, k])
        elif(index == "Bensaid"):
          metrics[1,k] = BensaidIndex(data, membership_matrix, centers)
        elif (index == "Kwon"):
          metrics[1,k] = KwonIndex(data, membership_matrix, centers)
        else:
          metrics[1,k] = TangIndex(data, membership_matrix, centers)

    return metrics

def compute_improvement_fuzzifier(validation_index):

    num_samples = 1000

    num_datasets = 9 * 19 * 15
    num_columns = 6

    col_names = ['num_centers', 'dim', 'id_m_2', 'best_m', 'id_best_m', 'gap']
    index_data = pd.DataFrame( np.zeros((num_datasets, num_columns)), columns = col_names)

    pos_m_2 = 49
    count = 0

    for dim in range(2, 21):
        for c in range(5, 11):
            for trials in range(2, 15):

                seed = 10000*c + 100*dim + trials

                np.random.seed(seed)
                centers = 2 * np.random.rand(c, dim) - 1

                seed *= 10

                np.random.seed(seed)
                std = 0.5 * np.random.rand(c)

                seed *= 10

                data, _ = make_blobs(n_samples = num_samples, n_features = dim,
                                    cluster_std = std, centers = centers,
                                    shuffle=False, random_state = seed)

                print("Dim: %d, Centers: %d, Trials: %d" %(dim, c, trials))
                metrics = evaluateFuzzifier(data, c, index = validation_index)

                index_data.iloc[count, 0] = c
                index_data.iloc[count, 1] = dim

                if (metrics.shape[1] <= pos_m_2):
                    index_data.iloc[count, 2] = np.nan
                else:
                    index_data.iloc[count, 2] = metrics[1, pos_m_2 ]

                pos_best_m = np.argmin( metrics[1, :] )

                index_data.iloc[count, 3] = metrics[0, pos_best_m]
                index_data.iloc[count, 4] = metrics[1, pos_best_m]

                index_data.iloc[count, 5] = np.abs( ( index_data.iloc[count, 2] - metrics[1, pos_best_m ] ) / metrics[1, pos_best_m ] )

                print(index_data.iloc[count].tolist())

                count += 1

    return index_data


compute_improvement_fuzzifier("Fukuyama-Sugeno")
