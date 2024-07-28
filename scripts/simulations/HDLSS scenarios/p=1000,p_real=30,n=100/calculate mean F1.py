import numpy as np
import pandas as pd

K_list = range(1, 21)
def calculate_F1(precision, recall):
    denominator = precision + recall
    F1_score = np.where(denominator != 0, 2 * (precision * recall) / denominator, 0)
    return F1_score

# F1 for Knoop ===========================================
precision_matrix_for_Knoop = np.load("precisions for knoop.npy")
average_precision_for_Knoop = np.mean(precision_matrix_for_Knoop, axis=0)
recall_matrix_for_Knoop = np.load("recalls for knoop.npy")
average_recall_for_Knoop = np.mean(recall_matrix_for_Knoop, axis=0)
average_F1_score_for_Knoop = calculate_F1(average_precision_for_Knoop, average_recall_for_Knoop)
result = {'K':K_list,
          'P for Knoop':average_precision_for_Knoop,
          'R for Knoop':average_recall_for_Knoop,
          'F1 for Knoop': average_F1_score_for_Knoop}
df = pd.DataFrame(result)
file_name = 'F1 results for Knoop.xlsx'
df.to_excel(file_name, index=False)

# F1 for Pearson ===========================================
precision_matrix_for_Pearson = np.load("precisions for Pearson.npy")
average_precision_for_Pearson = np.mean(precision_matrix_for_Pearson, axis=0)
recall_matrix_for_Pearson = np.load("recalls for Pearson.npy")
average_recall_for_Pearson = np.mean(recall_matrix_for_Pearson, axis=0)
average_F1_score_for_Pearson = calculate_F1(average_precision_for_Pearson, average_recall_for_Pearson)
result = {'K':K_list,
          'P for Pearson':average_precision_for_Pearson,
          'R for Pearson':average_recall_for_Pearson,
          'F1 for Pearson': average_F1_score_for_Pearson}
df = pd.DataFrame(result)
file_name = 'F1 results for Pearson.xlsx'
df.to_excel(file_name, index=False)

# F1 for Knockoff ===========================================
precision_matrix_for_Knockoff = np.load("precisions for Knockoff.npy")
average_precision_for_Knockoff = np.mean(precision_matrix_for_Knockoff, axis=0)
recall_matrix_for_Knockoff = np.load("recalls for Knockoff.npy")
average_recall_for_Knockoff = np.mean(recall_matrix_for_Knockoff, axis=0)
average_F1_score_for_Knockoff = calculate_F1(average_precision_for_Knockoff, average_recall_for_Knockoff)
result = {'K':K_list,
          'P for Knockoff':average_precision_for_Knockoff,
          'R for Knockoff':average_recall_for_Knockoff,
          'F1 for Knockoff': average_F1_score_for_Knockoff}
df = pd.DataFrame(result)
file_name = 'F1 results for Knockoff.xlsx'
df.to_excel(file_name, index=False)

# F1 for ridge regression ===========================================
precision_matrix_for_ridge = np.load("precisions for ridge.npy")
average_precision_for_ridge = np.mean(precision_matrix_for_ridge, axis=0)
recall_matrix_for_ridge = np.load("recalls for ridge.npy")
average_recall_for_ridge = np.mean(recall_matrix_for_ridge, axis=0)
average_F1_score_for_ridge = calculate_F1(average_precision_for_ridge, average_recall_for_ridge)
result = {'K':K_list,
          'P for ridge':average_precision_for_ridge,
          'R for ridge':average_recall_for_ridge,
          'F1 for ridge': average_F1_score_for_ridge}
df = pd.DataFrame(result)
file_name = 'F1 results for ridge.xlsx'
df.to_excel(file_name, index=False)

# F1 for lasso ===========================================
precision_matrix_for_lasso = np.load("precisions for lasso.npy")
average_precision_for_lasso = np.mean(precision_matrix_for_lasso, axis=0)
recall_matrix_for_lasso = np.load("recalls for lasso.npy")
average_recall_for_lasso = np.mean(recall_matrix_for_lasso, axis=0)
average_F1_score_for_lasso = calculate_F1(average_precision_for_lasso, average_recall_for_lasso)
result = {'K':K_list,
          'P for lasso':average_precision_for_lasso,
          'R for lasso':average_recall_for_lasso,
          'F1 for lasso': average_F1_score_for_lasso}
df = pd.DataFrame(result)
file_name = 'F1 results for lasso.xlsx'
df.to_excel(file_name, index=False)

# F1 for elasticNet===========================================
precision_matrix_for_elasticNet = np.load("precisions for elasticNet.npy")
average_precision_for_elasticNet = np.mean(precision_matrix_for_elasticNet, axis=0)
recall_matrix_for_elasticNet = np.load("recalls for elasticNet.npy")
average_recall_for_elasticNet = np.mean(recall_matrix_for_elasticNet, axis=0)
average_F1_score_for_elasticNet = calculate_F1(average_precision_for_elasticNet, average_recall_for_elasticNet)
result = {'K':K_list,
          'P for elasticNet':average_precision_for_elasticNet,
          'R for elasticNet':average_recall_for_elasticNet,
          'F1 for elasticNet': average_F1_score_for_elasticNet}
df = pd.DataFrame(result)
file_name = 'F1 results for elasticNet.xlsx'
df.to_excel(file_name, index=False)