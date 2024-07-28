import numpy as np
import pandas as pd
from sklearn import linear_model
import scipy.stats as stats
from knockoffBO_tools import get_X_matrix_with_multiple_knockoffs
from MSE_tools import load_datasets_for_GeneralizedLinearRegression
from get_everything_about_BestKFeatures_GBDT import get_indices_from_the_best_feature_alone_to_all_features, get_MSE_list_for_BestKFeatures
from parameters import data_dim, knockoff_hier

def get_datasets_for_knockoffBO_CV():
    all_X_train, all_y_train, X_test, y_test = load_datasets_for_GeneralizedLinearRegression(data_dim)
    print(all_y_train.shape)
    print(all_X_train.shape)
    X_train_part_1 = all_X_train[0:4, :]
    y_train_part_1 = all_y_train[0:4]
    X_train_part_2 = all_X_train[4:8, :]
    y_train_part_2 = all_y_train[4:8]
    X_train_part_3 = all_X_train[8:12, :]
    y_train_part_3 = all_y_train[8:12]
    X_train_part_4 = all_X_train[12:16, :]
    y_train_part_4 = all_y_train[12:16]
    X_train_part_5 = all_X_train[16:, ]
    y_train_part_5 = all_y_train[16:, ]
    # Cross-validation trial 1 - leave part5 as test set in this trail
    X_train_CV_1 = np.concatenate((X_train_part_1, X_train_part_2, X_train_part_3, X_train_part_4))
    y_train_CV_1 = np.concatenate((y_train_part_1, y_train_part_2, y_train_part_3, y_train_part_4))
    X_test_CV_1 = X_train_part_5
    y_test_CV_1 = y_train_part_5
    dataset_CV_1 = (X_train_CV_1, y_train_CV_1, X_test_CV_1, y_test_CV_1)
    # Cross-validation trial 2 - leave part 1 as test set in this trail
    X_train_CV_2 = np.concatenate((X_train_part_2, X_train_part_3, X_train_part_4, X_train_part_5))
    y_train_CV_2 = np.concatenate((y_train_part_2, y_train_part_3, y_train_part_4, y_train_part_5))
    X_test_CV_2 = X_train_part_1
    y_test_CV_2 = y_train_part_1
    dataset_CV_2 = (X_train_CV_2, y_train_CV_2, X_test_CV_2, y_test_CV_2)
    # Cross-validation trial 3 - leave part 2 as test set in this trail
    X_train_CV_3 = np.concatenate((X_train_part_1, X_train_part_3, X_train_part_4, X_train_part_5))
    y_train_CV_3 = np.concatenate((y_train_part_1, y_train_part_3, y_train_part_4, y_train_part_5))
    X_test_CV_3 = X_train_part_2
    y_test_CV_3 = y_train_part_2
    dataset_CV_3 = (X_train_CV_3, y_train_CV_3, X_test_CV_3, y_test_CV_3)
    # Cross-validation trial 4 - leave part 3 as test set in this trail
    X_train_CV_4 = np.concatenate((X_train_part_1, X_train_part_2, X_train_part_4, X_train_part_5))
    y_train_CV_4 = np.concatenate((y_train_part_1, y_train_part_2, y_train_part_4, y_train_part_5))
    X_test_CV_4 = X_train_part_3
    y_test_CV_4 = y_train_part_3
    dataset_CV_4 = (X_train_CV_4, y_train_CV_4, X_test_CV_4, y_test_CV_4)
    # Cross-validation trial 5 - leave part 4 as test set in this trail
    X_train_CV_5 = np.concatenate((X_train_part_1, X_train_part_2, X_train_part_3, X_train_part_5))
    y_train_CV_5 = np.concatenate((y_train_part_1, y_train_part_2, y_train_part_3, y_train_part_5))
    X_test_CV_5 = X_train_part_4
    y_test_CV_5 = y_train_part_4
    dataset_CV_5 = (X_train_CV_5, y_train_CV_5, X_test_CV_5, y_test_CV_5)
    return dataset_CV_1, dataset_CV_2, dataset_CV_3, dataset_CV_4, dataset_CV_5






def get_train_and_test_from_datasetCV(datasetCV):
    X_train_CV, y_train_CV, X_test_CV, y_test_CV = datasetCV
    return X_train_CV, y_train_CV, X_test_CV, y_test_CV



def get_knockoff_matrix_for_datasetCV(X_train_CV):
    print("Generating knockoff matrix for training features...")
    X_train_knockoff = get_X_matrix_with_multiple_knockoffs(X_train_CV)
    return X_train_knockoff




def do_ridgeless_regression_and_give_back_coefficients(X_train_knockoff, y_train):
    # ridgeless回归得到参数
    ridge_model = linear_model.Ridge() # alpha=1 之后可调
    ridge_model.fit(X_train_knockoff, y_train)
    return ridge_model.coef_


def put_coefficients_of_features_and_their_knockoffs_into_matrix(coefficientMatrix, coefficients):
    # Write down the correct value into the coefficient matrix.
    for fea_index in range(data_dim):
        for knockoff_index in range(2 ** knockoff_hier):
            # knockoff_index = 0 : real feature
            # knockoff_index = 1 ~ 2 ** knock_hier : real features' knockoffs
            total_index_in_the_coefficient_list = fea_index + knockoff_index * data_dim
            coefficientMatrix[fea_index][knockoff_index] = coefficients[total_index_in_the_coefficient_list]
    return coefficientMatrix


def get_coefficientMatrix_each_row_represents_a_feature(coefficients):
    """
    put coefficients into lists and return the lists as a matrix.
    The matrix is a list of list, each sub-list(or a row) corresponds to a feature
    The first value in a sub-list is coefficient with the feature itself,
    while the others are with its knockoffs.
    """
    coefficientMatrix_each_row_represents_a_feature = [[0 for j in range(2 ** knockoff_hier)] for i in range(data_dim)]
    print("How many features are we testing:",len(coefficientMatrix_each_row_represents_a_feature))
    print("How many knockoffs does a feature have:", len(coefficientMatrix_each_row_represents_a_feature[0]))
    coefficientMatrix_each_row_represents_a_feature = put_coefficients_of_features_and_their_knockoffs_into_matrix(coefficientMatrix_each_row_represents_a_feature, coefficients)
    return coefficientMatrix_each_row_represents_a_feature


def get_p_value_array(coefficient_Matrix):
    """
    Return the p-value array and save it to .npy file.
    """
    p_values = np.zeros(data_dim)
    p_values = put_values_into_p_value_array(p_values, coefficient_Matrix)
    return p_values


def put_values_into_p_value_array(p_valueArray, coefficient_Matrix):
    """
    Put values into the p-value list by each feature.
    :param p-value Array
    :return: p-value Array
    """
    for fea_index in range(data_dim):
        p_valueArray[fea_index] = calculate_p_value_by_Z_score_method_for_this_feature(fea_index, coefficient_Matrix)
    return p_valueArray


def calculate_p_value_by_Z_score_method_for_this_feature(fea_index, coefficient_Matrix):
    """
    return p-value for each feature
    :param fea_index:
    :return: p_value
    """
    feature_coefficient, knockoff_coefficients = get_coefficients_for_this_feature_and_its_knockoffs(fea_index, coefficient_Matrix)
    p_value = Z_score_method(feature_coefficient, knockoff_coefficients)
    return p_value


def get_coefficients_for_this_feature_and_its_knockoffs(feaIndex, coefficient_Matrix):
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



def get_p_value_list(X_train_CV, y_train_CV):
    X_train_knockoff_for_CV = get_knockoff_matrix_for_datasetCV(X_train_CV)
    coefficients = do_ridgeless_regression_and_give_back_coefficients(X_train_knockoff_for_CV, y_train_CV)
    coefficient_Matrix = get_coefficientMatrix_each_row_represents_a_feature(coefficients)
    p_value_list = get_p_value_array(coefficient_Matrix)
    return p_value_list


def get_best_K_index(dataset_CV):
    # 解压缩
    X_train_CV, y_train_CV, X_test_CV, y_test_CV = get_train_and_test_from_datasetCV(dataset_CV)
    # 计算各变量的p-value
    p_value_list = get_p_value_list(X_train_CV, y_train_CV)
    # 得到最佳索引列表清单
    list_of_bestK_indices_lists = get_indices_from_the_best_feature_alone_to_all_features(p_value_list)
    # 得到第K个最佳变量对应的MSE清单
    MSE_list = get_MSE_list_for_BestKFeatures(list_of_bestK_indices_lists)
    # 使用min()函数找到最小值
    min_MSE = min(MSE_list)
    # 使用index()方法找到最小值的索引
    best_K = MSE_list.index(min_MSE)

    return best_K, min_MSE

def save_bestK_list(bestK_list, mse_list):
    # 将最佳索引列表清单存入excel文件
    result = {'CV trial number': list(range(1, 6)),
              'best K-1': bestK_list,
              'mse': mse_list}
    df = pd.DataFrame(result)
    output_file = 'Result_for_CV_KnockoffBO - GBDT.xlsx'  # 设置输出的 Excel 文件名
    df.to_excel(output_file, index=False, sheet_name="CV")  # 将 DataFrame 写入 Excel，不保存索引列
    print("Best Ks are saved to excel.")


# ==========================================================================
dataset_CV_1, dataset_CV_2, dataset_CV_3, dataset_CV_4, dataset_CV_5 = get_datasets_for_knockoffBO_CV()

trial_number = 0
bestK_list = []
mse_list = []
for dataset_CV in get_datasets_for_knockoffBO_CV():
    trial_number += 1
    print("CV trial number:", trial_number)
    best_K, mse = get_best_K_index(dataset_CV)
    bestK_list.append(best_K)
    mse_list.append(mse)

save_bestK_list(bestK_list, mse_list)


