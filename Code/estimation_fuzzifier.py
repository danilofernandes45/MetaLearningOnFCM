#First Step
import numpy as np
from math import log10, exp, sqrt, floor
from sklearn.datasets import make_blobs
from sklearn.preprocessing import minmax_scale
from scipy.stats import kurtosis, skew
from matplotlib import pyplot as plt

#Compute Meta-Features ==============================================================================================
def computeDescriptors(vector):
    descriptors = np.zeros(19)
    descriptors[0] = vector.mean()
    descriptors[1] = vector.var()
    descriptors[2] = vector.std()
    descriptors[3] = skew(vector)
    descriptors[4] = kurtosis(vector)
    descriptors[5:15] = np.histogram(vector, bins = np.arange(0, 1.1, 0.1))[0] / vector.size

    abs_std_vector = abs( ( vector - vector.mean() ) / vector.std() )
    descriptors[15:19] = np.histogram( abs_std_vector, bins = [0, 1, 2, 3, np.inf])[0] / vector.size
    return descriptors

def metafeatures_based_on_dissimilarity_correlation(data):
    N, p = data.shape
    rank_data = data.argsort(axis = 1).argsort(axis = 1)
    spearman_rank_correlations = []
    for j in range(N):
        for k in range(j+1, N):
            correlation = 1 - 6*( ( rank_data[j] - rank_data[k] )**2 ).sum() / (p**3 - p)
            spearman_rank_correlations.append(correlation)

    distances = computeInnerDistances(data) ** 0.5
    dist_corr = np.append(spearman_rank_correlations, distances)
    normalized_dist_corr = ( dist_corr - dist_corr.min() ) / ( dist_corr.max() - dist_corr.min() )
    metafeatures = computeDescriptors(normalized_dist_corr)
    return metafeatures


def metafeatures_based_on_distances(data):
    distances = computeInnerDistances(data) ** 0.5
    normalized_dists = ( distances - distances.min() ) / ( distances.max() - distances.min() )
    metafeatures = computeDescriptors(normalized_dists)
    return metafeatures

def metafeatures_based_on_density(data, n_bins):
    N = data.shape[0]
    densities = []
    for r in range(N):
        dens = 0
        for s in range(N):
            dist = ( ( data[r] - data[s] )**2 ).sum()
            dens += exp(-dist/2)
        densities.append( dens / (N * sqrt(2*np.pi) ) )

    densities = np.array(densities)
    normalized_densities = ( densities - densities.min() ) / ( densities.max() - densities.min() )
    histogram = np.histogram(normalized_densities, bins = n_bins)[0]
    return histogram

#CLUSTERS EVALUATION ================================================================================================
def partitionCoefficient(membership_matrix): #(Bezdek, 1975)
    c, N = membership_matrix.shape
    sum = 0.0

    for i in range(c):
        for j in range(N):
            sum += membership_matrix[i, j]**2

    mean = sum / N
    return(mean)

def partitionEntropy(membership_matrix): #(Bezdek, 1974)
    c, N = membership_matrix.shape
    sum_entropy = 0.0

    for i in range(c):
        for j in range(N):
            if(membership_matrix[i,j] != 0):
                sum_entropy += membership_matrix[i, j] * log10(membership_matrix[i, j])

    mean_entropy = - sum_entropy / N
    return(mean_entropy)

def computeInnerDistances(vectors):
    c = vectors.shape[0]
    dists = []

    for i in range(c):
        for l in range(i + 1, c):
            dists.append( ( ( vectors[i] - vectors[l] )**2 ).sum() )

    return np.array( dists )

def XieBeniIndex(data, membership_matrix, centers, m): #(Xie and Beni, 1991) and modified by (Pal and Bezdek, 1995)

    N = data.shape[0]
    xb_index = computeObjetiveFunction(data, membership_matrix, centers, m) / ( N * computeInnerDistances(centers).min() )

    return xb_index

def BensaidIndex(data, membership_matrix, centers): #(Bensaid et. al. 1996)

    c, N = membership_matrix.shape
    bs_index = 0.0

    for i in range(c):
        sum_dist_to_centers = 0.0
        sum_dist_between_centers = 0.0
        for j in range(N):
            dist_data = ( ( data[j] - centers[i] )**2 ).sum()
            sum_dist_to_centers += ( membership_matrix[i, j]**2 ) * dist_data

        for k in range(c):
            sum_dist_between_centers += ( (centers[i] - centers[k])**2).sum()

        bs_index += sum_dist_to_centers / ( sum_dist_between_centers * np.sum(membership_matrix[i]) )

    return(bs_index)

def KwonIndex(data, membership_matrix, centers): #(Kwon et al, 1996)
    c = centers.shape[0]

    scaled_dist_sum = computeObjetiveFunction(data, membership_matrix, centers, 2)
    center_data = data.mean(axis = 0)
    mean_dist_to_center_data = ( (centers - center_data)**2 ).sum() / c

    kwon_index = ( scaled_dist_sum + mean_dist_to_center_data ) /  computeInnerDistances(centers).min()

    return kwon_index

def TangIndex(data, membership_matrix, centers): #(Tang et al, 2005)

    c = centers.shape[0]
    scaled_dist_sum = computeObjetiveFunction(data, membership_matrix, centers, 2)
    dist_inner_centers = computeInnerDistances(centers)

    tang_index = ( scaled_dist_sum + 2*dist_inner_centers.sum()/(c**2-c) ) / ( dist_inner_centers.min() + 1/c )

    return tang_index

def FukuyamaSugenoIndex(data, membership_matrix, centers, m): #(Fukuyama and Sugeno, 1989)

    mean_center = centers.mean(axis = 0)
    c, N = membership_matrix.shape
    scaled_dist_sum = 0.0

    dist_data = 0.0
    dist_centers = 0.0

    for i in range(c):
        for j in range(N):
            dist_data = ( (centers[i] - data[j])**2 ).sum()
            dist_centers = ( (centers[i] - mean_center)**2 ).sum()
            scaled_dist_sum += ( membership_matrix[i, j]**m ) * ( dist_data - dist_centers )

    return scaled_dist_sum

def ZahidIndex(data, membership_matrix, centers, m): #(Zahid et al, 1999)

    c, N = membership_matrix.shape
    mean_center = centers.mean(axis = 0)
    mean_dist_center = ( ( centers - mean_center )**2 ).sum() / c

    scaled_dist_sum = 0.0

    for i in range(c):

        dists_to_center_i =  ( ( data - centers[i] )**2 ).sum(axis = 1)
        scaled_dists_to_center_i = ( membership_matrix[i] ** m ) * dists_to_center_i
        scaled_dist_sum += scaled_dists_to_center_i.sum() / membership_matrix[i].sum()

    sc_1 = mean_dist_center / scaled_dist_sum

    min_sum = 0.0
    for i in range(c):
        for l in range(i+1, c):
            sum1 = 0.0
            sum2 = 0.0
            for j in range(N):
                sum1 += min( membership_matrix[i,j], membership_matrix[l,j] ) ** 2
                sum2 += min( membership_matrix[i,j], membership_matrix[l,j] )

            min_sum += sum1 / sum2

    ratio_max = ( membership_matrix.max(axis = 0) ** 2 ).sum() / ( membership_matrix.max(axis = 0) ).sum()

    sc_2 = min_sum / ratio_max

    return (sc_1 - sc_2)

def FuzzyHypervolumeIndex(data, membership_matrix, centers, m): #(Gath and Geva, 1989)

    c, N = membership_matrix.shape
    p = data.shape[1]
    index = 0.0
    for i in range(c):

        trans_data = data - centers[i]
        F_i = np.zeros([p,p])

        for j in range(N):
            vector = np.array([trans_data[j]])
            F_i = F_i + ( membership_matrix[i,j] ** m) * vector.transpose().dot(vector)

        F_i = F_i / ( membership_matrix[i] ** m ).sum()

        det = np.linalg.det(F_i)
        if(det < 0):
            det = 0
        index += det**0.5

    return index

def PCAES_Index(data, membership_matrix, centers): #(Wu and Yang, 2005)

    c, N = membership_matrix.shape
    center_data = data.mean(axis = 0)
    beta = ( ( centers - center_data ) ** 2 ).sum() / c
    u_m =  ( membership_matrix ** 2 ).sum(axis = 1).min()

    index = 0.0
    for i in range(c):

        index += ( membership_matrix[i]**2 ).sum() / u_m

        rm_centers = np.delete(centers, i, axis = 0)
        dist_between_centers = ( ( rm_centers - centers[i] )**2 ).sum(axis = 1) / beta
        index -= exp( -dist_between_centers.min() )

    return index

def SCG_Index(data, membership_matrix, centers, m): #(Bouguessa and Wang, 2004)

    c, N = membership_matrix.shape
    p = data.shape[1]
    globalCompactness = 0.0

    for i in range(c):

        trans_data = data - centers[i]
        F_i = np.zeros([p,p])

        for j in range(N):
            vector = np.array([trans_data[j]])
            F_i = F_i + ( membership_matrix[i,j] ** m) * vector.transpose().dot(vector)

        F_i = F_i / ( membership_matrix[i] ** m ).sum()

        globalCompactness += F_i.trace()

    trans_centers = centers - centers.mean(axis=0)
    S_b = np.zeros([p,p])

    for i in range(c):
        vector = np.array([trans_centers[i]])
        S_b = S_b + ( vector.transpose().dot(vector) * ( membership_matrix[i]**m ).sum() )

    separation = S_b.trace()

    return ( separation / globalCompactness )

def AverageWithinClusterDistance(data, membership_matrix, centers, m):

    c, N = membership_matrix.shape

    index = 0.0
    for i in range(c):
         dists_to_center_i = ( ( data - centers[i] )**2 ).sum(axis = 1)
         scaled_dist_sum = ( ( membership_matrix[i]**m ) * dists_to_center_i ).sum()
         index += scaled_dist_sum / ( membership_matrix[i]**m ).sum()

    index = index / (c * N)
    return index

#FUZZY C-MEANS ======================================================================================================

def updateMembership(data, centers, m):
    N = data.shape[0]
    c = centers.shape[0]
    membership_matrix = np.zeros([c, N])

    for j in range(N):
        for i in range(c):
            dist = ( ( data[j] - centers[i] )**2 ).sum()
            if( dist == 0 ):
                membership_matrix[:,j] = 0
                membership_matrix[i,j] = 1
                break
            membership_matrix[i, j] = dist ** (-1 / (m - 1))
            if(membership_matrix[i, j] > 1e300):
                membership_matrix[:,j] = 0
                membership_matrix[i,j] = 1
                break
        membership_matrix[:, j] /= membership_matrix[:, j].sum()

    return membership_matrix

def updateCenters(data, membership_matrix, c, m):
    centers = (membership_matrix**m).dot(data)
    for i in range(c):
        centers[i] /= ( membership_matrix[i] ** m ).sum()

    return centers

def computeObjetiveFunction(data, membership_matrix, centers, m):
    N = data.shape[0]
    c = centers.shape[0]
    sum = 0

    for j in range(N):
        for i in range(c):
            sum += ( membership_matrix[i, j] ** m ) * ( ( data[j] - centers[i] ) ** 2 ).sum()

    return sum

def fuzzyCMeans(data, c, m, eps = 0.001, seed = None, max_iter = 1000):
    #VARIABLES
    N, p = data.shape
    centers = np.zeros([c, p])
    #RANDOM INITIALIZE OF MEMBERSHIP
    np.random.seed(seed)
    membership_matrix = np.random.random(N*c).reshape([c, N])
    for j in range(N):
        membership_matrix[:, j] /= membership_matrix[:, j].sum()
    #LOOP VARIABLES
    k = 0
    fun_obj1 = 0
    fun_obj2 = 0
    #OBTAINING MEMBERSHIP MATRIX AND CENTERS
    while( k < 3 or ( abs(fun_obj2 - fun_obj1) >= eps and k < max_iter ) ):
        centers = updateCenters(data, membership_matrix, c, m)
        membership_matrix = updateMembership(data, centers, m)
        fun_obj1 = fun_obj2
        fun_obj2 = computeObjetiveFunction(data, membership_matrix, centers, m)
        k += 1

    return (membership_matrix, centers, abs(fun_obj2 - fun_obj1), k)

#FUZZIFIER ESTIMATION ==============================================================================================

def maxTheoreticalFuzzifier(data):

    N, p = data.shape
    std_data = data - np.mean(data, axis = 0)
    C_x = np.zeros([p, p])

    for j in range(N):
        vector = np.array([std_data[j]])
        C_x = C_x + vector.transpose().dot(vector) / ( N * (vector**2).sum() )

    max_eigenvalue = max(np.linalg.eigvals(C_x))
    max_m = np.inf

    if(max_eigenvalue < 0.5):
        max_m = 1 / (1 - 2 * max_eigenvalue )
    print(max_m)
    return max_m

def evaluateFuzzifier(data, c):

    max_m = min(50, maxTheoreticalFuzzifier(data))

    m_values = np.arange(1.02, max_m, 0.02)
    metrics = np.zeros([11, m_values.size])
    metrics[0] = m_values

    for k in range(m_values.size):
        membership_matrix, centers, error, num_iter = fuzzyCMeans(data, c, metrics[0, k], seed = 123)

        metrics[1,k] = ZahidIndex(data, membership_matrix, centers, metrics[0, k])
        metrics[2,k] = PCAES_Index(data, membership_matrix, centers)
        metrics[3,k] = SCG_Index(data, membership_matrix, centers, metrics[0,k])

        metrics[4,k] = FukuyamaSugenoIndex(data, membership_matrix, centers, metrics[0, k])
        metrics[5,k] = XieBeniIndex(data, membership_matrix, centers, metrics[0, k])
        metrics[6,k] = BensaidIndex(data, membership_matrix, centers)
        metrics[7,k] = KwonIndex(data, membership_matrix, centers)
        metrics[8,k] = TangIndex(data, membership_matrix, centers)
        metrics[9,k] = FuzzyHypervolumeIndex(data, membership_matrix, centers, metrics[0,k])
        metrics[10,k] = AverageWithinClusterDistance(data, membership_matrix, centers, metrics[0,k])

    return metrics

#TEST =====================================================================================================

def generateMetadatasets():

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

            metrics = evaluateFuzzifier(X, c)
            print("%d - %d - %d"%(id, p, c))
            for k in range(1, 11):
                idx = 0
                if(k <= 3):
                    idx = np.argwhere( metrics[k] == metrics[k].max() )[0,0]
                else:
                    idx = np.argwhere( metrics[k] == metrics[k].min() )[0,0]
                estimate_m = np.append( estimate_m, metrics[0, idx] )

    mf_density = mf_density.reshape([54, mean_n_samples]) #size = 234
    mf_distances = mf_distances.reshape([54, 19])
    mf_dist_corr = mf_dist_corr.reshape([54, 19])
    estimate_m = estimate_m.reshape([54, 10])

    np.savetxt("mf_density_simulated.csv", mf_density, delimiter = ",", fmt="%.5f")
    np.savetxt("mf_distances_simulated.csv", mf_distances, delimiter = ",", fmt="%.5f")
    np.savetxt("mf_dist_corr_simulated.csv", mf_dist_corr, delimiter = ",", fmt="%.5f")
    np.savetxt("estimate_m_simulated.csv", estimate_m, delimiter = ",", fmt="%.5f")


generateMetadatasets()

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
