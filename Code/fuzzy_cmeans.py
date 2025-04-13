import numpy as np
from math import log10, exp, sqrt, floor
from sklearn.datasets import make_blobs
from sklearn.preprocessing import minmax_scale
from scipy.stats import kurtosis, skew
# from matplotlib import pyplot as plt

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.metrics import pairwise_distances

#CONSTANTS
NUM_STATS = 19
NUM_INDICES = 10

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
    dists = computeInnerDistances(centers)
    min_dist_centers = dists[dists != 0].min()
    xb_index = np.inf
    if(min_dist_centers > 0):
        xb_index = computeObjetiveFunction(data, membership_matrix, centers, m) / ( N * min_dist_centers )
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
        
        sum_membership = np.sum(membership_matrix[i])
        if sum_membership == 0:
            sum_membership = 1e-10

        bs_index += sum_dist_to_centers / ( sum_dist_between_centers * sum_membership )

    return min(bs_index, 1e10)

def KwonIndex(data, membership_matrix, centers): #(Kwon et al, 1996)
    c = centers.shape[0]

    scaled_dist_sum = computeObjetiveFunction(data, membership_matrix, centers, 2)
    center_data = data.mean(axis = 0)
    mean_dist_to_center_data = ( (centers - center_data)**2 ).sum() / c

    min_dist_centers = computeInnerDistances(centers).min()
    kwon_index = np.inf
    if( min_dist_centers > 0 ):
        kwon_index = ( scaled_dist_sum + mean_dist_to_center_data ) /  min_dist_centers

    return min(kwon_index, 1e10)

def TangIndex(data, membership_matrix, centers): #(Tang et al, 2005)

    c = centers.shape[0]
    scaled_dist_sum = computeObjetiveFunction(data, membership_matrix, centers, 2)
    dist_inner_centers = computeInnerDistances(centers)

    tang_index = ( scaled_dist_sum + 2*dist_inner_centers.sum()/(c**2-c) ) / ( dist_inner_centers.min() + 1/c )

    return min(tang_index, 1e10)

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

    return min(scaled_dist_sum[0], 1e10)

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
            if(membership_matrix[i, j] > 1e10):
                membership_matrix[:,j] = 0
                membership_matrix[i,j] = 1
                break
        membership_matrix[:, j] /= membership_matrix[:, j].sum()

    return membership_matrix

def updateCenters(data, membership_matrix, c, m):
    centers = (membership_matrix**m).dot(data)
    for i in range(c):
        u_sum = ( membership_matrix[i] ** m ).sum()
        if u_sum > 0:
            centers[i] /= u_sum

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

    max_eigenvalue = max(np.real(np.linalg.eigvals(C_x)))
    max_m = np.inf

    if(max_eigenvalue < 0.5):
        max_m = 1 / (1 - 2 * max_eigenvalue )
    print(max_m)
    return max_m