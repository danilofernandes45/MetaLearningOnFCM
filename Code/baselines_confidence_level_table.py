import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

T_VALUE = 2.0452 # T-VALUE for df = 29 and alpha = 0.05 (confidence level equal to 0.95)
N = 30 #Number of samples

indices = ['B','FS','K','T']

# avg_mape = pd.read_csv("../Data/Synthetic/Error_Datasets/Baselines/average_mape_error.csv")
# med_mape = pd.read_csv("../Data/Synthetic/Error_Datasets/Baselines/median_mape_error.csv")
#
# avg_rrmse = pd.read_csv("../Data/Synthetic/Error_Datasets/Baselines/average_rrmse_error.csv")
# med_rrmse = pd.read_csv("../Data/Synthetic/Error_Datasets/Baselines/median_rrmse_error.csv")

avg_mape = pd.read_csv("../Data/Real/Error_Datasets/Baselines/average_mape_error.csv")
med_mape = pd.read_csv("../Data/Real/Error_Datasets/Baselines/median_mape_error.csv")

avg_rrmse = pd.read_csv("../Data/Real/Error_Datasets/Baselines/average_rrmse_error.csv")
med_rrmse = pd.read_csv("../Data/Real/Error_Datasets/Baselines/median_rrmse_error.csv")

baselines = [avg_mape, med_mape, avg_rrmse, med_rrmse]
col1 = ["MAPE", "MAPE", "RRMSE", "RRMSE"]
col2 = ["Average", "Median", "Average", "Median"]

print("\\begin{table}[hbt]")
print("    \\centering")
print("    \\begin{tabular}{c|c|c|c|c|c}")
print("        \\multicolumn{2}{*}{} & Bensaid & Fukuyama-Sugeno & Kwon & Tang\\\\")
print("        \\hline")
for i in range(4):
    data = baselines[i][indices]
    data_mean = data.mean()
    data_var = data.var()
    max = ( data_mean + T_VALUE * np.sqrt(data_var / N) ).round(4)
    min = ( data_mean - T_VALUE * np.sqrt(data_var / N) ).round(4)

    print("        "+col1[i]+" & "+col2[i]+" & [%.4f, %.4f] & [%.4f, %.4f] & [%.4f, %.4f] & [%.4f, %.4f]\\\\"%(min['B'], max['B'], min['FS'], max['FS'], min['K'], max['K'], min['T'], max['T']))

print("    \\end{tabular}")
print("    \\caption{Baselines Confidence Interval for real datasets}")
print("    \\label{tab:real_baselines_confidence}")
print("\\end{table}")
print()
