import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

indices_names = ["Z_Index", "PCAES_Index", "SCG_Index", "FS_Index", "XB_Index", "B_Index", "K_Index", "T_Index", "FH_Index", "AWCD_Index"]
metafeatures_names = ["Metafeature_Density", "Metafeature_Distance", "Metafeature_Dist_Cor"]

indices = ['Z','PCAES','SCG','FS','XB','B','K','T','FH','AWCD']

labels = ["K = 1", "K = 2", "K = 3", "K = 4", "K = 5", "K = 6", "K = 7", "K = 8", "K = 9", "K = 10", "Average", "Marjority"]

avg_mape = pd.read_csv("Error_Datasets/average_mape_error.csv")
marj_mape = pd.read_csv("Error_Datasets/marjority_mape_error.csv")

avg_rrmse = pd.read_csv("Error_Datasets/average_rrmse_error.csv")
marj_rrmse = pd.read_csv("Error_Datasets/marjority_rrmse_error.csv")


for i in range(10):
    for m in metafeatures_names:
        mape = pd.read_csv("Error_Datasets/"+m+"/"+indices_names[i]+"/mape.csv", header = None)
        rrmse = pd.read_csv("Error_Datasets/"+m+"/"+indices_names[i]+"/rrmse.csv", header = None)

        print("Error_Datasets/"+m+"/"+indices_names[i]+"/mape.csv")

        data_mape = pd.concat([mape, avg_mape[indices[i]], marj_mape[indices[i]]], axis = 1)
        print(data_mape)
        plt.boxplot(data_mape.T, labels = labels, notch = True)
        plt.show()

        print("Error_Datasets/"+m+"/"+indices_names[i]+"/rrmse.csv")

        data_rrmse = pd.concat([rrmse, avg_rrmse[indices[i]], marj_rrmse[indices[i]]], axis = 1)
        plt.boxplot(data_rrmse.T, labels = labels, notch = True)
        plt.show()
