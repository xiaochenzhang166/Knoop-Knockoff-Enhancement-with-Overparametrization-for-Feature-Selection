p = 1000  # 变量个数
p_real = 30 # 真实模型中分量不为零的变量个数
err_in_beta = 0 #beta中无关变量的系数
n = 3000 # 训练样本个数
n_test = 300 # 测试样本个数
err = 0.25 # 真实模型中的误差项的方差
knock_hier = 3 # knockoffs hierarchy numbers
alpha = 0.01 # 正则化强度，可根据需要调整
rho = 0.1
repeat_times = 20


import numpy as np
import pandas as pd
from numpy import random
import sklearn.preprocessing as prep
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import scipy.stats as stats
import statsmodels.api as sm
from knockpy.knockoff_filter import KnockoffFilter
from sklearn import metrics


def generate_sigma(ndim):
    global sigma
    # generate covariate matrix Σ
    sigma = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            sigma[i, j] = rho ** abs(i - j)

def create_Beta(ndim: int, n_real_dim):
    global beta
    np.random.seed(42)  # 设置随机种子
    # 随机产生前n_real_dim个分量
    primative_beta = np.random.uniform(0, 1, (1, n_real_dim))
    beta = prep.normalize(primative_beta, axis=1).T
    # 给其余分量补零
    zeros_to_add = np.zeros((ndim - n_real_dim, 1))
    noises_to_add = np.full_like(zeros_to_add, err_in_beta)
    # 使用np.vstack将两个数组垂直堆叠
    beta = np.vstack((noises_to_add, beta))
    beta = random.permutation(beta)
    np.save("real_beta.npy", beta)


def normalize_rows(matrix):
    """
    归一化矩阵的每一行，使其L2范数为1。
    参数：    matrix (numpy.ndarray)：输入的矩阵。
    返回：    normalized_matrix (numpy.ndarray)：每一行都被归一化为L2范数为1的矩阵。
    """
    # 计算每一行的L2范数
    row_norms = np.linalg.norm(matrix, axis=1)
    # 归一化每一行，使其L2范数为1
    normalized_matrix = matrix / row_norms[:, np.newaxis]
    return normalized_matrix

def generate_Xandy(ndim, nsample,knockoff_hierarchy,seed,train=True):
    """
    生成数值实验用到的全部变量
    :param ndim: 变量维数
    :param nsample: 样本点个数
    :param knockoff_hierarchy: 做多少次knockoffs生成
    :param seed: 随机数种子
    :return: 包含knockoff的设计矩阵的扩展，根据真实模型生成的y
    """
    np.random.seed(seed)  # Set the random seed
    X = np.random.multivariate_normal(np.zeros(ndim), sigma, nsample)
    y = X @ beta + err * np.random.randn(nsample, 1)
    y = y.reshape(-1)
    if train == True:
        np.save('original_X_train.npy', X)
        np.save('y_train.npy', y)

    for i in tqdm(range(knockoff_hierarchy)):
        X = get_knockoffs(X, y)
    normalized_X = normalize_rows(X) # normalize matrix by row (every l2 norm of a row is 1)
    return normalized_X, y


def get_knockoffs(X, y):
    # Run model-X knockoffs
    kfilter = KnockoffFilter(
        fstat='ridge',
        ksampler='gaussian',
    )
    rejections = kfilter.forward(X=X, y=y)
    X = np.hstack((X, kfilter.Xk))
    return X


def ridge_mse():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    train_mse = []
    test_mse = []
    for i in tqdm(range(knock_hier)):
        ndim = p * 2 ** (i + 1)

        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train[:, :ndim], y_train)
        train_residuals = y_train - ridge_model.predict(X_train[:, :ndim])
        training_error = np.mean(train_residuals ** 2)
        train_mse.append(training_error)

        y_pred = ridge_model.predict(X_test[:, :ndim])
        mse = mean_squared_error(y_test, y_pred)
        test_mse.append(mse)

    return train_mse, test_mse






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

# Precision and Recall ================
def calculate_precision_recall(scores, true_labels):
    precision_values = []
    recall_values = []

    for k in range(1, 21):
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        actual_positives = sum(1 for idx in top_k_indices if true_labels[idx] == 1)
        total_positives = sum(1 for label in true_labels if label == 1)

        precision = actual_positives / k
        recall = actual_positives / total_positives if total_positives > 0 else 0

        precision_values.append(precision)
        recall_values.append(recall)

    return precision_values, recall_values





# main code ==========================================================
# Initialize the lists ---------------------------------------
auc_list_for_Knoop = []
auc_list_for_Pearson = []
auc_list_for_linear_regression = []
auc_list_for_Knockoff = []
auc_list_for_ridge_regression = []
auc_list_for_lasso = []
auc_list_for_elasticNet = []

fpr_value_lists_for_Knoop = []
tpr_value_lists_for_Knoop = []
fpr_value_lists_for_Knockoff = []
tpr_value_lists_for_Knockoff = []
fpr_value_lists_for_Pearson = []
tpr_value_lists_for_Pearson = []
fpr_value_lists_for_ridge = []
tpr_value_lists_for_ridge = []
fpr_value_lists_for_lasso = []
tpr_value_lists_for_lasso = []
fpr_value_lists_for_elasticNet = []
tpr_value_lists_for_elasticNet = []

precision_value_lists_for_Knoop = []
recall_value_lists_for_Knoop = []
precision_value_lists_for_Pearson = []
recall_value_lists_for_Pearson = []
precision_value_lists_for_Knockoff = []
recall_value_lists_for_Knockoff = []
precision_value_lists_for_ridge = []
recall_value_lists_for_ridge = []
precision_value_lists_for_lasso = []
recall_value_lists_for_lasso = []
precision_value_lists_for_elasticNet = []
recall_value_lists_for_elasticNet = []

# -------------------------------------------------------------------------
###################
# For Loop Begins #
###################
# -------------------------------------------------------------------------

for seed in tqdm(range(repeat_times)):
    generate_sigma(p)
    create_Beta(p, p_real)

    X_train, y_train = generate_Xandy(p, n, knock_hier, seed)
    X_test, y_test = generate_Xandy(p, n_test, knock_hier, 42, train=False)
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # load data ==================================================================
    beta = np.load('real_beta.npy')
    label = np.where(beta != 0, 1, 0)

    X_train_for_knoop = np.load('X_train.npy')
    X_train = np.load('original_X_train.npy')
    y_train = np.load('y_train.npy')

    #  Knoop Procedures ==========================================
    ridge_model_for_Knoop = Ridge(alpha=alpha)
    ridge_model_for_Knoop.fit(X_train_for_knoop, y_train)
    coefficient_Matrix = detectionMatrix(knock_hier, ridge_model_for_Knoop)
    p_values_array = get_p_value_array()
    p_scores_array_for_Knoop = get_p_scores_array_for_Knoop(p_values_array)
    # print(p_scores_array_for_Knoop)

    # Pearson Correlation Coefficient Methods =================================================
    Pearson_coefficients = []
    # 遍历X的每一列（每个变量），计算相关系数
    for column in X_train.T:
        correlation_coefficient = np.corrcoef(column, y_train)[0, 1]
        Pearson_coefficients.append(correlation_coefficient)
    Pearson_coefficients_abs = np.abs(Pearson_coefficients)
    # print("Pearson correlatioin coefficients are:", Pearson_coefficients_abs)

    # # Linear Regression p-values Method ===========================================
    # p_scores_for_regression = []
    # # 遍历X的每一列（每个变量），进行线性回归并计算p-value
    # for column in X_train.T:
    #     X_constant = sm.add_constant(column)  # 添加常数项
    #     model = sm.OLS(y_train, X_constant).fit()  # 拟合线性回归模型
    #     p_value_for_regression = model.pvalues[1]  # 获取变量的p-value
    #     p_scores_for_regression.append(1 - p_value_for_regression)
    # # print("p_scores_for_regression:",p_scores_for_regression)

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
    # print("ridge coefficients:", ridge_coefficients_abs)

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

    # calculate metrics =================================================
    # calculate metrics for Knoop ------------------------------------------
    Knoop_fpr = calculate_fpr(p_scores_array_for_Knoop, label)
    # print("False Positive Rate:",Knoop_fpr)
    Knoop_tpr = calculate_tpr(p_scores_array_for_Knoop, label)
    # print("True positive rate:", Knoop_tpr)
    roc_auc_for_Knoop = metrics.auc(Knoop_fpr, Knoop_tpr)
    # print(roc_auc_for_Knoop)
    precision_values_for_Knoop, recall_values_for_Knoop\
        = calculate_precision_recall(p_scores_array_for_Knoop, label)

    # calculate metric for Pearson Correlation -----------------------------
    Pearson_fpr = calculate_fpr(Pearson_coefficients_abs, label)
    Pearson_tpr = calculate_tpr(Pearson_coefficients_abs, label)
    roc_auc_for_Pearson = metrics.auc(Pearson_fpr, Pearson_tpr)
    precision_values_for_Pearson, recall_values_for_Pearson \
        = calculate_precision_recall(Pearson_coefficients_abs, label)


    # calculate metric for Knockoff ---------------------------------------
    Knockoff_fpr = calculate_fpr(W_score_array, label)
    Knockoff_tpr = calculate_tpr(W_score_array, label)
    roc_auc_for_Knockoff = metrics.auc(Knockoff_fpr, Knockoff_tpr)
    precision_values_for_Knockoff, recall_values_for_Knockoff \
        = calculate_precision_recall(W_score_array, label)

    # calculate metric for Ridge Regression -----------------------------
    ridge_regression_fpr = calculate_fpr(ridge_coefficients_abs, label)
    ridge_regression_tpr = calculate_tpr(ridge_coefficients_abs, label)
    roc_auc_for_ridge_regression = metrics.auc(ridge_regression_fpr, ridge_regression_tpr)
    precision_values_for_ridge, recall_values_for_ridge \
        = calculate_precision_recall(ridge_coefficients_abs, label)

    # calculate metric for Lasso Regression -----------------------------
    lasso_regression_fpr = calculate_fpr(lasso_coefficients_abs, label)
    lasso_regression_tpr = calculate_tpr(lasso_coefficients_abs, label)
    roc_auc_for_lasso_regression = metrics.auc(lasso_regression_fpr, lasso_regression_tpr)
    precision_values_for_lasso, recall_values_for_lasso \
        = calculate_precision_recall(lasso_coefficients_abs, label)

    # calculate metric for ElasticNet Regression ------------------------
    elasticNet_regression_fpr = calculate_fpr(elastic_net_abs, label)
    elasticNet_regression_tpr = calculate_tpr(elastic_net_abs, label)
    roc_auc_for_elasticNet_regression = metrics.auc(elasticNet_regression_fpr, elasticNet_regression_tpr)
    precision_values_for_elasticNet, recall_values_for_elasticNet \
        = calculate_precision_recall(elastic_net_abs, label)


    auc_list_for_Knoop.append(roc_auc_for_Knoop)
    auc_list_for_Pearson.append(roc_auc_for_Pearson)
    auc_list_for_Knockoff.append(roc_auc_for_Knockoff)
    auc_list_for_ridge_regression.append(roc_auc_for_ridge_regression)
    auc_list_for_lasso.append(roc_auc_for_lasso_regression)
    auc_list_for_elasticNet.append(roc_auc_for_elasticNet_regression)

    fpr_value_lists_for_Knoop.append(Knoop_fpr)
    tpr_value_lists_for_Knoop.append(Knoop_tpr)
    fpr_value_lists_for_Knockoff.append(Knockoff_fpr)
    tpr_value_lists_for_Knockoff.append(Knockoff_tpr)
    fpr_value_lists_for_Pearson.append(Pearson_fpr)
    tpr_value_lists_for_Pearson.append(Pearson_tpr)
    fpr_value_lists_for_ridge.append(ridge_regression_fpr)
    tpr_value_lists_for_ridge.append(ridge_regression_tpr)
    fpr_value_lists_for_lasso.append(lasso_regression_fpr)
    tpr_value_lists_for_lasso.append(lasso_regression_tpr)
    fpr_value_lists_for_elasticNet.append(elasticNet_regression_fpr)
    tpr_value_lists_for_elasticNet.append(elasticNet_regression_tpr)

    precision_value_lists_for_Knoop.append(precision_values_for_Knoop)
    recall_value_lists_for_Knoop.append(recall_values_for_Knoop)
    precision_value_lists_for_Pearson.append(precision_values_for_Pearson)
    recall_value_lists_for_Pearson.append(recall_values_for_Pearson)
    precision_value_lists_for_Knockoff.append(precision_values_for_Knockoff)
    recall_value_lists_for_Knockoff.append(recall_values_for_Knockoff)
    precision_value_lists_for_ridge.append(precision_values_for_ridge)
    recall_value_lists_for_ridge.append(recall_values_for_ridge)
    precision_value_lists_for_lasso.append(precision_values_for_lasso)
    recall_value_lists_for_lasso.append(recall_values_for_lasso)
    precision_value_lists_for_elasticNet.append(precision_values_for_elasticNet)
    recall_value_lists_for_elasticNet.append(recall_values_for_elasticNet)


# save the results ============================
# save AUC ----------------------------------
result = {'Knoop':auc_list_for_Knoop,
          'Pearson':auc_list_for_Pearson,
          'Knockoff':auc_list_for_Knockoff,
          'Ridge':auc_list_for_ridge_regression,
          'Lasso':auc_list_for_lasso,
          'ElasticNet':auc_list_for_elasticNet
          }
df = pd.DataFrame(result)
file_name = 'AUC results.xlsx'  # 设置输出的 Excel 文件名
df.to_excel(file_name, index=False)


# save fpr and tpf -------------------------------
np.save("Fpr values for knoop.npy", fpr_value_lists_for_Knoop)
np.save("Tpr values for knoop.npy", tpr_value_lists_for_Knoop)
np.save("Fpr values for Pearson.npy", fpr_value_lists_for_Pearson)
np.save("Tpr values for Pearson.npy", tpr_value_lists_for_Pearson)
np.save("Fpr values for knockoff.npy", fpr_value_lists_for_Knockoff)
np.save("Tpr values for knockoff.npy", tpr_value_lists_for_Knockoff)
np.save("Fpr values for ridge.npy", fpr_value_lists_for_ridge)
np.save("Tpr values for ridge.npy", tpr_value_lists_for_ridge)
np.save("Fpr values for lasso.npy", fpr_value_lists_for_lasso)
np.save("Tpr values for lasso.npy", tpr_value_lists_for_lasso)
np.save("Fpr values for elasticNet.npy", fpr_value_lists_for_elasticNet)
np.save("Tpr values for elasticNet.npy", tpr_value_lists_for_elasticNet)

# save Precision and Recalls -----------------------
np.save("precisions for knoop.npy", precision_value_lists_for_Knoop)
np.save("recalls for knoop.npy", recall_value_lists_for_Knoop)
np.save("precisions for Pearson.npy", precision_value_lists_for_Pearson)
np.save("recalls for Pearson.npy", recall_value_lists_for_Pearson)
np.save("precisions for Knockoff.npy", precision_value_lists_for_Knockoff)
np.save("recalls for Knockoff.npy", recall_value_lists_for_Knockoff)
np.save("precisions for ridge.npy", precision_value_lists_for_ridge)
np.save("recalls for ridge.npy", recall_value_lists_for_ridge)
np.save("precisions for lasso.npy", precision_value_lists_for_lasso)
np.save("recalls for lasso.npy", recall_value_lists_for_lasso)
np.save("precisions for elasticNet.npy", precision_value_lists_for_elasticNet)
np.save("recalls for elasticNet.npy", recall_value_lists_for_elasticNet)


