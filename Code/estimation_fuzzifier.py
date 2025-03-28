#First Step
import pandas as pd
from fuzzy_cmeans import *
from evaluate_fuzzifier import *

#TEST =====================================================================================================

def generateSyntheticMetadatasets():

    mf_density = np.array([])
    mf_distances = np.array([])
    mf_dist_corr = np.array([])
    estimate_m = np.array([])

    np.random.seed(123)
    n_samples = np.random.choice([200, 300, 400, 500, 600], 234)
    mean_n_samples = floor( n_samples.mean() + 1 )
    id = 180 # min = 0
    for p in range(25, 31): #min 5, max = 31
        np.random.seed(123)
        centers = 10 * np.random.random(10*p).reshape([10,p])
        np.random.seed(123)
        std = 5 * np.random.random(10*p).reshape([10,p])
        for c in range(2, 11):
            X, _ = make_blobs(n_samples = n_samples[id], n_features = p, cluster_std = std[0:c], centers = centers[0:c], shuffle=False, random_state = 123)
            X = minmax_scale(X)
            id += 1

            mf_density = np.append( mf_density, metafeatures_based_on_density(X, mean_n_samples) )
            mf_distances = np.append( mf_distances, metafeatures_based_on_distances(X) )
            mf_dist_corr = np.append( mf_dist_corr, metafeatures_based_on_dissimilarity_correlation(X) )

            metrics = gridSearch(X, c)
            print("%d - %d - %d"%(id, p, c))
            for k in range(1, 11):
                idx = 0
                if(k <= 3):
                    idx = np.argwhere( metrics[k] == np.nanmax(metrics[k]) )[0,0]
                else:
                    idx = np.argwhere( metrics[k] == np.nanmin(metrics[k]) )[0,0]
                estimate_m = np.append( estimate_m, metrics[0, idx] )

    mf_density = mf_density.reshape([54, mean_n_samples]) #size = 234
    mf_distances = mf_distances.reshape([54, 19])
    mf_dist_corr = mf_dist_corr.reshape([54, 19])
    estimate_m = estimate_m.reshape([54, 10])

    np.savetxt("mf_density_simulated.csv", mf_density, delimiter = ",", fmt="%.5f")
    np.savetxt("mf_distances_simulated.csv", mf_distances, delimiter = ",", fmt="%.5f")
    np.savetxt("mf_dist_corr_simulated.csv", mf_dist_corr, delimiter = ",", fmt="%.5f")
    np.savetxt("estimate_m_simulated.csv", estimate_m, delimiter = ",", fmt="%.5f")

def generateMetadatasets():
    mf_density = np.array([])
    mf_distances = np.array([])
    mf_dist_corr = np.array([])
    estimate_m = np.array([])

    desc_datasets = pd.read_csv("../../Real_Datasets_OpenML/desc_datasets.csv")

    mean_n_samples = floor( desc_datasets['Num_Rows'].mean() + 1 )

    for i in range(desc_datasets.shape[0]):

        df = pd.read_csv("../../Real_Datasets_OpenML/"+desc_datasets['Name'][i])
        p = desc_datasets['Num_Columns'][i]
        c = df.iloc[:,p].nunique() #Number of classes
        X = df.iloc[:,:p].to_numpy() #Datasets without targets
        X = minmax_scale(X)

        mf_density = np.append( mf_density, metafeatures_based_on_density(X, mean_n_samples) )
        mf_distances = np.append( mf_distances, metafeatures_based_on_distances(X) )
        mf_dist_corr = np.append( mf_dist_corr, metafeatures_based_on_dissimilarity_correlation(X) )

        metrics = gridSearch(X, c)
        print("%d - %d - %d"%(i, p, c))
        for k in range(1, 11):
            idx = 0
            if(k <= 3):
                idx = np.argwhere( metrics[k] == np.nanmax(metrics[k]) )[0,0]
            else:
                idx = np.argwhere( metrics[k] == np.nanmin(metrics[k]) )[0,0]
            estimate_m = np.append( estimate_m, metrics[0, idx] )

    size = desc_datasets.shape[0]
    mf_density = mf_density.reshape([size, mean_n_samples])
    mf_distances = mf_distances.reshape([size, NUM_STATS])
    mf_dist_corr = mf_dist_corr.reshape([size, NUM_STATS])
    estimate_m = estimate_m.reshape([size, NUM_INDICES])

    np.savetxt("mf_density_real.csv", mf_density, delimiter = ",", fmt="%.5f")
    np.savetxt("mf_distances_real.csv", mf_distances, delimiter = ",", fmt="%.5f")
    np.savetxt("mf_dist_corr_real.csv", mf_dist_corr, delimiter = ",", fmt="%.5f")
    np.savetxt("estimate_m_real.csv", estimate_m, delimiter = ",", fmt="%.5f")


# Bayesian Optimization ======================================

def estimateSynthetic():

    estimate_m = np.array([])

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
            id += 1

            metrics = bayesianSearch(X, c)
            print("%d - %d - %d"%(id, p, c))
            estimate_m = np.append( estimate_m, metrics )

    estimate_m = estimate_m.reshape([234, 4])
    np.savetxt("bayes_opt_m_simulated.csv", estimate_m, delimiter = ",", fmt="%.5f")

def estimateReal():

    estimate_m = np.array([])
    desc_datasets = pd.read_csv("../Data/Real/Real_Datasets_OpenML/desc_datasets.csv")
    size = desc_datasets.shape[0]

    for i in range(size):

        df = pd.read_csv("../Data/Real/Real_Datasets_OpenML/"+desc_datasets['Name'][i])
        p = desc_datasets['Num_Columns'][i]
        c = df.iloc[:,p].nunique() #Number of classes
        X = df.iloc[:,:p].to_numpy() #Datasets without targets
        X = minmax_scale(X)

        metrics = bayesianSearch(X, c)
        print("%d - %d - %d"%(i, p, c))
        estimate_m = np.append( estimate_m, metrics )

    estimate_m = estimate_m.reshape([size, 4])

    np.savetxt("bayes_opt_m_real.csv", estimate_m, delimiter = ",", fmt="%.5f")

estimateReal()

#CLUSTERING PLOTS

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(xs = X[:,0], ys = X[:,1], zs = X[:,2])
# plt.show()
#
# plt.scatter(X[:,0], X[:,1])
# plt.show()
# plt.scatter(X[:,0], X[:,2])
# plt.show()
# plt.scatter(X[:,1], X[:,2])
# plt.show()

#np.random.seed(12345)
#centers = 10 * np.random.random(6).reshape([3,2])
#np.random.seed(12345)
#std = 3 * np.random.random(6).reshape([3,2])
# centers = [(0,0), (-1,0),(1,0)]
# std = [1,0.25,0.25]
# X, _ = make_blobs(n_samples = 500, n_features = 2, cluster_std = std, centers = centers, shuffle=False, random_state = 12345)
#
# print( ( centers - X.min() ) / (X.max() - X.min()))
#
# X = minmax_scale(X)


# metrics = evaluateFuzzifier(X, 3)

# for k in range(1, 11):
#     idx = 0
#     if(k <= 3):
#         idx = np.argwhere( metrics[k] == metrics[k].max() )[0,0]
#     else:
#         idx = np.argwhere( metrics[k] == metrics[k].min() )[0,0]
#     print( metrics[0, idx] )


# membership_matrix, centers, error, num_iter = fuzzyCMeans(X, 3, 1.84, seed = 12345)
# print(centers)
#
# fig, axs = plt.subplots(2,2)
# color = np.zeros([500,4])
# color[:,0] = 1
# color[:,3] = membership_matrix[0]
#
# axs[0,0].scatter(X[:,0], X[:,1])
# axs[0,0].set_title("Data (m = 2)")
#
# axs[0,1].scatter(X[:,0], X[:,1], color = color)
# axs[0,1].set_title(centers[0])
#
# color[:,0] = 0
# color[:,1] = 1
# color[:,3] = membership_matrix[1]
#
# axs[1,0].scatter(X[:,0], X[:,1], color = color)
# axs[1,0].set_title(centers[1])
#
# color[:,1] = 0
# color[:,2] = 1
# color[:,3] = membership_matrix[2]
#
# axs[1,1].scatter(X[:,0], X[:,1], color = color)
# axs[1,1].set_title(centers[2])
# plt.show()
