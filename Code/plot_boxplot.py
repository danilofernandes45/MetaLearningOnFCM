import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

indices_names = ["Z_Index", "PCAES_Index", "SCG_Index", "FS_Index", "XB_Index", "B_Index", "K_Index", "T_Index", "FH_Index", "AWCD_Index"]
metafeatures_names = ["Metafeature_Density", "Metafeature_Distance", "Metafeature_Dist_Cor"]

indices = ['Z','PCAES','SCG','FS','XB','B','K','T','FH','AWCD']

#labels = ["K = 1", "K = 2", "K = 3", "K = 4", "K = 5", "K = 6", "K = 7", "K = 8", "K = 9", "K = 10", "Average", "Marjority"]
#labels = ["K = 1", "K = 2", "K = 3", "K = 4", "K = 5", "K = 6", "K = 7", "K = 8", "K = 9", "K = 10", "Average", "Median"]

# labels = []
# for gamma in ["0.1", "1", "10", "10²", "10³", "10⁴"]:
#     for c in ["0.1", "1", "10", "10²", "10³", "10⁴"]:
#         labels.append("C = "+c+"\ngamma="+gamma)
# labels.append("Average")
# labels.append("Median")
# labels = np.array(labels)

labels = np.array(["N = 10", "N = 20", "N = 40", "N = 60", "N = 80", "N = 100", "N = 200", "N = 300", "N = 400", "N = 500", "Average", "Median"])

avg_mape = pd.read_csv("../Data/Real/Error_Datasets/Baselines/average_mape_error.csv")
# marj_mape = pd.read_csv("../Data/Error_Datasets/marjority_mape_error.csv")
med_mape = pd.read_csv("../Data/Real/Error_Datasets/Baselines/median_mape_error.csv")

avg_rrmse = pd.read_csv("../Data/Real/Error_Datasets/Baselines/average_rrmse_error.csv")
#marj_rrmse = pd.read_csv("../DataError_Datasets/marjority_rrmse_error.csv")
med_rrmse = pd.read_csv("../Data/Real/Error_Datasets/Baselines/median_rrmse_error.csv")


for i in range(10):
    for m in metafeatures_names:
        #mape = pd.read_csv("../Data/Real/Error_Datasets/knn/"+m+"/"+indices_names[i]+"_mape.csv", header = None)
        #rrmse = pd.read_csv("../Data/Real/Error_Datasets/knn/"+m+"/"+indices_names[i]+"_rrmse.csv", header = None)

        # mape = pd.read_csv("../Data/Real/Error_Datasets/svm_rbf/"+m+"/"+indices_names[i]+"_mape.csv", header = None)
        # rrmse = pd.read_csv("../Data/Real/Error_Datasets/svm_rbf/"+m+"/"+indices_names[i]+"_rrmse.csv", header = None)

        mape = pd.read_csv("../Data/Real/Error_Datasets/random_forest/"+m+"/"+indices_names[i]+"_mape.csv", header = None)
        rrmse = pd.read_csv("../Data/Real/Error_Datasets/random_forest/"+m+"/"+indices_names[i]+"_rrmse.csv", header = None)

        print("../Data/Real/Error_Datasets/random_forest/"+m+"/"+indices_names[i]+"_mape.csv")

        data_mape = pd.concat([mape, avg_mape[indices[i]], med_mape[indices[i]]], axis = 1)
        #print(data_mape)
        plt.boxplot([vals.dropna() for col, vals in data_mape.iteritems()], labels = labels, notch = True)
        plt.show()

        # for w in [12, 24, 36]:
        #     data_mape = pd.concat([mape.iloc[:, (w-12):w], avg_mape[indices[i]], med_mape[indices[i]]], axis = 1)
        #     #print(data_mape)
        #     xlabel = np.append(labels[(w-12):w], labels[36:])
        #     plt.boxplot([vals.dropna() for col, vals in data_mape.iteritems()], labels = xlabel, notch = True)
        #     plt.show()

        print("../Data/Real/Error_Datasets/random_forest/"+m+"/"+indices_names[i]+"_rrmse.csv")

        data_rrmse = pd.concat([rrmse, avg_rrmse[indices[i]], med_rrmse[indices[i]]], axis = 1)
        plt.boxplot([vals.dropna() for col, vals in data_rrmse.iteritems()], labels = labels, notch = True)
        plt.show()

        # for w in [12, 24, 36]:
        #     data_rrmse = pd.concat([rrmse.iloc[:, (w-12):w], avg_rrmse[indices[i]], med_rrmse[indices[i]]], axis = 1)
        #     xlabel = np.append(labels[(w-12):w], labels[36:])
        #     plt.boxplot([vals.dropna() for col, vals in data_rrmse.iteritems()], labels = xlabel, notch = True)
        #     plt.show()
