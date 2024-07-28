import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from knockpy.knockoff_filter import KnockoffFilter
from all import p, alpha, p_real, knock_hier
from sklearn import metrics

# load data ==================================================================
beta = np.load('real_beta.npy')
label = np.where(beta != 0, 1, 0)
print(label)

X_train_for_knoop = np.load('X_train.npy')
X_train = np.load('original_X_train.npy')
y_train = np.load('y_train.npy')

# Knoop Method ===================================================================
# Knoop Functions-------------------------------------------------------------
def detectionMatrix(knock_hier, ridge_model):
    detection_matrix = [[0 for j in range(2 ** knock_hier)] for i in range(p)]
    for var in range(p):
        for index in range(2 ** knock_hier):
            detection_matrix[var][index] = ridge_model.coef_[var + index * p]

    return detection_matrix

def Z_score_method(featureCoefficient, knockoffCoefficients):
    """
    检验feature变量的ridgeless回归系数，
    是否和其knockoff变量的ridgeless回归系数来自同一个正态总体，
    返回检验的p值.
    H0: 是同一总体
    H1: 不是
    """

    # 假设knockoff变量的ridgeless回归系数服从某一个正态分布
    # 计算分布的均值和样本方差，它们完全确定了正态分布
    mean = np.mean(knockoffCoefficients)
    std_dev = np.std(knockoffCoefficients) # standard deviation

    # 计算feature变量的系数落在该分布的哪一处（cdf）
    cdf = stats.norm.cdf(featureCoefficient, loc=mean, scale=std_dev)

    # 计算落在这个边缘区间的概率，即p-value
    p_value = 0.5 - abs(0.5 - cdf)

    return p_value

def calculate_p_value_by_Z_score_method_for_this_feature(fea_index):
    """
    return p-value for each feature
    :param fea_index:
    :return: p_value
    """
    feature_coefficient, knockoff_coefficients = get_coefficients_for_this_feature_and_its_knockoffs(fea_index)
    p_value = Z_score_method(feature_coefficient, knockoff_coefficients)
    return p_value

def get_coefficients_for_this_feature_and_its_knockoffs(feaIndex):
    """
    Extract a features' coefficient and its knockoffs' coefficients from coe-Matrix.
    :param feaIndex:
    :return: feature_coefficient, knockoff_coefficients
    """

    # Matrix的第index行代表第index（从0开始）个feature
    coefficients_of_this_feature_and_its_knockoffs = coefficient_Matrix[feaIndex]
    # 第0列是feature本身的ridgeless系数
    feature_coefficient = coefficients_of_this_feature_and_its_knockoffs[0]
    # 剩下的列是其knockoff的ridgeless系数
    knockoff_coefficients = coefficients_of_this_feature_and_its_knockoffs[1:]
    return feature_coefficient, knockoff_coefficients

def get_p_value_array():
    """
    Return the p-value array and save it to .npy file.
    """
    p_values = np.zeros(p)
    p_values = put_values_into_p_value_array(p_values)
    np.save("p-values.npy", p_values)
    print("p-values are saved as p-values.npy file.(Array type)")
    return p_values

def put_values_into_p_value_array(p_valueArray):
    """
    Put values into the p-value list by each feature.
    :param p-value Array
    :return: p-value Array
    """
    for fea_index in range(p):
        p_valueArray[fea_index] = calculate_p_value_by_Z_score_method_for_this_feature(fea_index)
    return p_valueArray

def get_p_scores_array_for_Knoop(p_values_array):
    p_scores_array = np.zeros(p)
    for var in range(p):
        p_scores_array[var] = 1 - p_values_array[var]
    return p_scores_array

#  Knoop Procedures ----------------------------------
ridge_model_for_Knoop = Ridge(alpha=alpha)
ridge_model_for_Knoop.fit(X_train_for_knoop, y_train)
coefficient_Matrix = detectionMatrix(knock_hier, ridge_model_for_Knoop)
p_values_array = get_p_value_array()
p_scores_array_for_Knoop = get_p_scores_array_for_Knoop(p_values_array)
print(p_scores_array_for_Knoop)





# Pearson Correlation Coefficient Methods =================================================
Pearson_coefficients = []
# 遍历X的每一列（每个变量），计算相关系数
for column in X_train.T:
    correlation_coefficient = np.corrcoef(column, y_train)[0, 1]
    Pearson_coefficients.append(correlation_coefficient)
Pearson_coefficients_abs = np.abs(Pearson_coefficients)
# print("Pearson correlatioin coefficients are:", Pearson_coefficients_abs)


# Linear Regression p-values Method ===========================================
p_scores_for_regression = []
# 遍历X的每一列（每个变量），进行线性回归并计算p-value
for column in X_train.T:
    X_constant = sm.add_constant(column)  # 添加常数项
    model = sm.OLS(y_train, X_constant).fit()  # 拟合线性回归模型
    p_value_for_regression = model.pvalues[1]  # 获取变量的p-value
    p_scores_for_regression.append(1-p_value_for_regression)
# print("p_scores_for_regression:",p_scores_for_regression)


# Knockoff ==========================================================
kfilter = KnockoffFilter(
    fstat='ridge',
    ksampler='gaussian',
)
rejections, W_score_array = kfilter.forward(X=X_train, y=y_train)
# print("W scores:", W_score_array)


# Ridge =============================================================
# 创建并拟合Ridge回归模型
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
# 将 Ridge 回归模型的系数取绝对值
ridge_coefficients_abs = np.abs(ridge_model.coef_)
print("ridge coefficients:", ridge_coefficients_abs)


# Lasso ============================================================
# 创建并拟合Lasso回归模型
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
# 将 Lasso 回归模型的系数取绝对值
lasso_coefficients_abs = np.abs(lasso_model.coef_)
# print("lasso coefficients:", lasso_coefficients_abs)


# ElasticNet ======================================================
# 创建并拟合ElasticNet模型
elastic_net_model = ElasticNet(random_state=0)
elastic_net_model.fit(X_train, y_train)
elastic_net_abs = np.abs(elastic_net_model.coef_)
# print("elastic net coefficients:", elastic_net_abs)


# fpr, tpr, auc =================================================
def calculate_fpr(scores, labels):
    # 将分数从高到低排列，并同时按照排列顺序调整标签
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    num_useful = np.sum(sorted_labels)  # 有用变量的个数
    num_useless = len(labels) - num_useful  # 无用变量的个数

    fpr_array = []  # 存储不同阈值下的 FPR

    num_selected_useless = 0  # 已选择的无用变量个数

    for i in range(len(labels)):
        if sorted_labels[i] == 0:
            num_selected_useless += 1

        fpr = num_selected_useless / num_useless
        fpr_array.append(fpr)

    return np.array(fpr_array)

# True positive rate
def calculate_tpr(scores, labels):
    # 将分数从高到低排列，并同时按照排列顺序调整标签
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    num_useful = np.sum(sorted_labels)  # 有用变量的个数

    tpr_array = []  # 存储不同阈值下的 TPR

    num_selected_useful = 0  # 已选择的有用变量个数

    for i in range(len(labels)):
        if sorted_labels[i] == 1:
            num_selected_useful += 1

        tpr = num_selected_useful / num_useful if num_useful > 0 else 0  # 防止除零错误
        tpr_array.append(tpr)

    return np.array(tpr_array)


# calculate metrics for Knoop ------------------------------------------
Knoop_fpr = calculate_fpr(p_scores_array_for_Knoop, label)
# print("False Positive Rate:",Knoop_fpr)
Knoop_tpr = calculate_tpr(p_scores_array_for_Knoop, label)
# print("True positive rate:", Knoop_tpr)
roc_auc_for_Knoop = metrics.auc(Knoop_fpr, Knoop_tpr)
# print(roc_auc_for_Knoop)

# calculate metric for Knockoff ---------------------------------------
Knockoff_fpr = calculate_fpr(W_score_array, label)
Knockoff_tpr = calculate_tpr(W_score_array, label)
roc_auc_for_Knockoff = metrics.auc(Knockoff_fpr, Knockoff_tpr)

# calculate metric for Ridge Regression -----------------------------
ridge_regression_fpr = calculate_fpr(ridge_coefficients_abs, label)
ridge_regression_tpr = calculate_tpr(ridge_coefficients_abs, label)
roc_auc_for_ridge_regression = metrics.auc(ridge_regression_fpr, ridge_regression_tpr)

# calculate metric for Lasso Regression -----------------------------
lasso_regression_fpr = calculate_fpr(lasso_coefficients_abs, label)
lasso_regression_tpr = calculate_tpr(lasso_coefficients_abs, label)
roc_auc_for_lasso_regression = metrics.auc(lasso_regression_fpr, lasso_regression_tpr)

# calculate metric for ElasticNet Regression ------------------------
elasticNet_regression_fpr = calculate_fpr(elastic_net_abs, label)
elasticNet_regression_tpr = calculate_tpr(elastic_net_abs, label)
roc_auc_for_elasticNet_regression = metrics.auc(elasticNet_regression_fpr, elasticNet_regression_tpr)


# plot ROC curves==========================================
plt.plot(Knoop_fpr, Knoop_tpr, color='red', lw=2, label='Knoop ROC curve (AUC = {:.2f})'.format(roc_auc_for_Knoop))
# plt.plot(Pearson_fpr, Pearson_tpr, color='darkorange', lw=2, label='Pearson Correlation ROC curve (AUC = {:.2f})'.format(roc_auc_for_Pearson))
# plt.plot(linear_regression_fpr, linear_regression_tpr, color='blue', lw=2, label='Linear Regression ROC curve (AUC = {:.2f})'.format(roc_auc_for_linear_regression))
plt.plot(Knockoff_fpr, Knockoff_tpr, color='green', lw=2, label='Knockoff ROC curve (AUC = {:.2f})'.format(roc_auc_for_Knockoff))
plt.plot(ridge_regression_fpr, ridge_regression_tpr, color='green', lw=2, label='Ridge Regression ROC curve (AUC = {:.2f})'.format(roc_auc_for_ridge_regression))
plt.plot(lasso_regression_fpr, lasso_regression_tpr, color='purple', lw=2, label='Lasso Regression ROC curve (AUC = {:.2f})'.format(roc_auc_for_lasso_regression))
plt.plot(elasticNet_regression_fpr, elasticNet_regression_tpr, color='magenta', lw=2, label='Elastic net ROC curve (AUC = {:.2f})'.format(roc_auc_for_elasticNet_regression))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('ROC curve with comparisons.png')
plt.show()
