import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

T_VALUE = 2.0452 # T-VALUE for df = 29 and alpha = 0.05 (confidence level equal to 0.95)
N = 30 #Number of sampless

indices_names = ["B_Index", "FS_Index", "K_Index", "T_Index"]
metafeatures_names = ["Metafeature_Density", "Metafeature_Distance", "Metafeature_Dist_Cor"]

indices = ['B','FS','K','T']

for id in range(4):
    for error in ["mape", "rrmse"]:
        table = [["$K$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$", "$10$"], [" & Density"], [" & Distance"], [" & Distance \& Correlation"] ]
        # table = [["$N$", "$10$", "$20$", "$40$", "$60$", "$80$", "$100$", "$200$", "$300$", "$400$", "$500$"], [" & Density"], [" & Distance"], [" & Distance \& Correlation"] ]
        # table = [["$\\gamma$", "\\multirow{5}{*}{$0.1$}", "", "", "", "",
        #                  "\\multirow{5}{*}{$1$}", "", "", "", "",
        #                  "\\multirow{5}{*}{$10$}", "", "", "", "",
        #                  "\\multirow{5}{*}{$10^2$}", "", "", "", "",
        #                  "\\multirow{5}{*}{$10^3$}", "", "", "", ""],
        #          [" & $C$", " & $0.1$", " & $1$", " & $10$", " & $10^2$", " & $10^3$",
        #                           " & $0.1$", " & $1$", " & $10$", " & $10^2$", " & $10^3$",
        #                           " & $0.1$", " & $1$", " & $10$", " & $10^2$", " & $10^3$",
        #                           " & $0.1$", " & $1$", " & $10$", " & $10^2$", " & $10^3$",
        #                           " & $0.1$", " & $1$", " & $10$", " & $10^2$", " & $10^3$",
        #                           " & $0.1$", " & $1$", " & $10$", " & $10^2$", " & $10^3$"],
        #          [" & Density"], [" & Distance"], [" & Distance \& Correlation"] ]

        for m in range(3):
            data = pd.read_csv("../Data/Real/Error_Datasets/knn/"+metafeatures_names[m]+"/"+indices_names[id]+"_"+error+".csv", header = None)
            data_mean = data.mean()
            data_var = data.var()
            max = ( data_mean + T_VALUE * np.sqrt(data_var / N) ).round(4)
            min = ( data_mean - T_VALUE * np.sqrt(data_var / N) ).round(4)

            # for i in range(1, 31):
            #     if(i%6 == 0):
            #         continue
            #     table[m+2].append(" & $[%.4f, %.4f]$"%(min[i-1], max[i-1]))

            for i in range(10):
                table[m+1].append(" & $[%.4f, %.4f]$"%(min[i], max[i]))

        print("\\begin{table}[hbt]")
        print("    \\centering")
        print("    \\begin{tabular}{c|c|c|c}")
        # for i in range(26):
        #     print("        "+table[0][i]+table[1][i]+table[2][i]+table[3][i]+table[4][i]+"\\\\")
        #     if(i == 0 or i%5 == 0 and i!=25):
        #         print("        \\hline")
        for i in range(11):
            print("        "+table[0][i]+table[1][i]+table[2][i]+table[3][i]+"\\\\")
            if(i == 0):
                     print("        \\hline")
        print("    \\end{tabular}")
        print("    \\caption{Real -- "+indices[id]+" -- "+error+"}")
        print("    \\label{tab:real_knn_"+indices[id]+"_"+error+"}")
        print("\\end{table}")
        print()
