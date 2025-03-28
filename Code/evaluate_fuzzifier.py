from skfuzzy.cluster import cmeans
from fuzzy_cmeans import *

# Função objetivo para otimização de m

def gridSearch(data, c):

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



def bayesianSearch(data, c):

    max_m = min(50, maxTheoreticalFuzzifier(data))

    params = {
        "data": data, 
        "c": c,  
        "seed": 123
    }

    fcm_func = lambda m : fuzzyCMeans(**params, m=m[0])[:2]
    func_indices = [
        # lambda m_space : list(map(lambda m : BensaidIndex(data,  *fcm_func(m)), m_space)),
        lambda m : BensaidIndex(data,  *fcm_func(m)),
        lambda m : FukuyamaSugenoIndex(data, *fcm_func(m), m),
        lambda m : KwonIndex(data, *fcm_func(m) ),
        lambda m : TangIndex(data, *fcm_func(m) )
    ]

    # Definição do espaço de busca para m
    space = [Real(1.02, max_m, name='m')]

    # Otimização Bayesiana
    bayes_opt = lambda func : gp_minimize(func, space, n_calls=20, random_state=123).x[0]
    best_m = list(map(bayes_opt, func_indices))
    print(best_m)
    return best_m