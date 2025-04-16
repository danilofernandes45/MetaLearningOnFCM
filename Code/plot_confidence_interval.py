import numpy as np
import matplotlib.pyplot as plt

def plot_conf_interval(median, avg, best_model, legend):

    colors = [
        ['#1560bd', '#73c2fb'], #Bensaid
        ['#d30000', '#fa8072'], #Fukuyama-Sugeno
        ['#2e8b57', '#d0f0c0'], #Kwon
        ['#ed7014', '#ffc594'] #Tang
    ]

    labels = ["Bensaid et al.", "Fukuyama-Sugeno", "Kwon et al.", "Tang et al."]

    for i in range(4):
        y_means = [
            ( avg[i][1] + avg[i][0] ) / 2,
            ( median[i][1] + median[i][0] ) / 2,
            ( best_model[i][1] + best_model[i][0] ) / 2
        ]

        y_range = [
            ( avg[i][1] - avg[i][0] ) / 2,
            ( median[i][1] - median[i][0] ) / 2,
            ( best_model[i][1] - best_model[i][0] ) / 2
        ]

        plt.errorbar(x = ["Average"+" "*i, "Median"+" "*i, legend[i]],
                        y = y_means, yerr = y_range, fmt = '.',
                        color = colors[i][0], ecolor = colors[i][1], elinewidth = 2,
                        capsize = 7, capthick = 2, label = labels[i])

        plt.errorbar(x = ["Average"+" "*i, "Median"+" "*i, legend[i]],
                        y = y_means, yerr = y_range, fmt = '.',
                        color = colors[i][0], ecolor = colors[i][1], elinewidth = 2.5,
                        capsize = 25, capthick = 2.5)

    plt.legend()
    plt.show()


#Synthetic Datasets

median_mape = [
    [0.0378,0.0379], #Bensaid
    [0.1072,0.1076], #Fukuyama-Sugeno
    [0.0484,0.0485], #Kwon
    [0.0735,0.0737] #Tang
]

median_rrmse = [
    [1.0258,1.0305], #Bensaid
    [1.0723,1.0751], #Fukuyama-Sugeno
    [1.0175,1.0238], #Kwon
    [1.0437,1.0481] #Tang
]

avg_mape = [
    [0.0389,0.0390], #Bensaid
    [0.2990,0.2996], #Fukuyama-Sugeno
    [0.0488,0.0489], #Kwon
    [0.0797,0.0798] #Tang
]

avg_rrmse = [
    [1.0212,1.0287], #Bensaid
    [1.0395,1.3275], #Fukuyama-Sugeno
    [1.0215,1.0292], #Kwon
    [1.0212,1.0319] #Tang
]

#CRITERIA: LOWEREST INTERVAL MEAN

best_model_mape = [
    [0.0258, 0.0260], #Bensaid => Dist&Cor + KNN ( K = 10 )
    [0.1285, 0.1291], #Fukuyama-Sugeno => Dist + SVM ( C = 0.1, gamma = 1 )
    [0.0425, 0.0428], #Kwon => Dist + SVM ( C = 0.1, gamma = 10 )
    [0.0421, 0.0425] #Tang => Dist&Cor + KNN (K = 9)
]

legend_mape = [
    "Dist. & Cor.\n + KNN \n (K = 10)",
    "Distance \n + SVM \n (C = 0.1, γ = 1)",
    "Distance \n + SVM \n (C = 0.1, γ = 10)",
    "Dist. & Cor. \n + KNN \n (K = 9)"
]

best_model_rrmse = [
    [0.6691, 0.6872], #Bensaid => Dist&Cor + KNN ( K = 10 )
    [0.7541, 0.8770], #Fukuyama-Sugeno => Dist + SVM (C = 1, gamma = 10)
    [0.8456, 0.8571], #Kwon => Dist + SVM ( C = 10, gamma = 1 )
    [0.5997, 0.6080] #Tang => Dist&Cor + SVM ( C = 10, gamma = 0.1 )
]

legend_rrmse = [
    "Dist. & Cor. \n + KNN \n (K = 10)",
    "Distance \n + SVM \n (C = 1, γ = 10)",
    "Distance \n + SVM \n (C = 10, γ = 1)",
    "Dist. & Cor. \n + SVM \n (C = 10, γ = 0.1)"
]

# plot_conf_interval(median_mape, avg_mape, best_model_mape, legend_mape)
# plot_conf_interval(median_rrmse, avg_rrmse, best_model_rrmse, legend_rrmse)

real_median_mape = [
    [0.1812, 0.1821], #Bensaid
    [0.4152, 0.4173], #Fukuyama-Sugeno
    [0.1641, 0.1650], #Kwon
    [0.1795, 0.1802]  #Tang
]

real_median_rrmse = [
    [1.0373, 1.0538], #Bensaid
    [1.0701, 1.0755], #Fukuyama-Sugeno
    [1.0607, 1.0728], #Kwon
    [1.0307, 1.0461]  #Tang
]

real_avg_mape = [
    [0.1915, 0.1920], #Bensaid
    [3.2385, 3.2468], #Fukuyama-Sugeno
    [0.1775, 0.1779], #Kwon
    [0.1837, 0.1841]  #Tang
]

real_avg_rrmse = [
    [1.0247, 1.0427], #Bensaid
    [1.1996, 1.4886], #Fukuyama-Sugeno
    [1.0243, 1.0409], #Kwon
    [1.0218, 1.0348]  #Tang
]

#CRITERIA: LOWEREST INTERVAL MEAN

real_best_model_mape = [
    [0.0917, 0.0926], #Bensaid
    [0.4014, 0.4042], #Fukuyama-Sugeno
    [0.0935, 0.0943], #Kwon
    [0.0824, 0.0833]  #Tang
]

real_legend_mape = [
    "Distance\n + KNN \n (K = 10)", #Bensaid
    "Distance\n + SVM \n (C = 0.1, γ = 10)", #Fukuyama-Sugeno
    "Distance\n + KNN \n (K = 8)", #Kwon
    "Dist. & Cor.\n + KNN \n (K = 9)"  #Tang
]

real_best_model_rrmse = [
    [0.6467, 0.6632], #Bensaid
    [1.0460, 1.0609], #Fukuyama-Sugeno
    [0.6525, 0.6656], #Kwon
    [0.5764, 0.5854]  #Tang
]

real_legend_rrmse = [
    "Distance\n + KNN \n (K = 8)", #Bensaid
    "Distance\n + SVM \n (C = 1, γ = 10²)", #Fukuyama-Sugeno
    "Distance\n + SVM \n (C = 10, γ = 0.1)", #Kwon
    "Distance\n + SVM \n (C = 10, γ = 0.1) "  #Tang
]

plot_conf_interval(real_median_mape, real_avg_mape, real_best_model_mape, real_legend_mape)
plot_conf_interval(real_median_rrmse, real_avg_rrmse, real_best_model_rrmse, real_legend_rrmse)
