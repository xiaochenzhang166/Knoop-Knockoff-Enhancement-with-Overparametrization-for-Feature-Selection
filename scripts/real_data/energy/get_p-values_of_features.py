import numpy as np
from sklearn import linear_model
from parameters import knockoff_hier, data_dim
import scipy.stats as stats


def load_data_for_ridgeless():
    X_train_knockoff = np.load("X_train_knockoff.npy",allow_pickle=True)
    y_train = np.load("y_train.npy",allow_pickle=True)
    return X_train_knockoff, y_train


def do_ridgeless_regression_and_give_back_coefficients():
    # ridgeless回归得到参数
    ridge_model = linear_model.Ridge() # alpha=1 之后可调
    ridge_model.fit(X_train_knockoff, y_train)
    return ridge_model.coef_


def put_coefficients_of_features_and_their_knockoffs_into_matrix(coefficientMatrix):
    # Write down the correct value into the coefficient matrix.
    for fea_index in range(data_dim):
        for knockoff_index in range(2 ** knockoff_hier):
            # knockoff_index = 0 : real feature
            # knockoff_index = 1 ~ 2 ** knock_hier : real features' knockoffs
            total_index_in_the_coefficient_list = fea_index + knockoff_index * data_dim
            coefficientMatrix[fea_index][knockoff_index] = coefficients[total_index_in_the_coefficient_list]
    return coefficientMatrix


def get_coefficientMatrix_each_row_represents_a_feature():
    """
    put coefficients into lists and return the lists as a matrix.
    The matrix is a list of list, each sub-list(or a row) corresponds to a feature
    The first value in a sub-list is coefficient with the feature itself,
    while the others are with its knockoffs.
    """
    coefficientMatrix_each_row_represents_a_feature = [[0 for j in range(2 ** knockoff_hier)] for i in range(data_dim)]
    print("How many features are we testing:",len(coefficientMatrix_each_row_represents_a_feature))
    print("How many knockoffs does a feature have:", len(coefficientMatrix_each_row_represents_a_feature[0]))
    coefficientMatrix_each_row_represents_a_feature = put_coefficients_of_features_and_their_knockoffs_into_matrix(coefficientMatrix_each_row_represents_a_feature)
    return coefficientMatrix_each_row_represents_a_feature


def get_p_value_array():
    """
    Return the p-value array and save it to .npy file.
    """
    p_values = np.zeros(data_dim)
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
    for fea_index in range(data_dim):
        p_valueArray[fea_index] = calculate_p_value_by_Z_score_method_for_this_feature(fea_index)
    return p_valueArray


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

# --------------------------------------------------------------
print("Testing and generating p-value list...")
X_train_knockoff, y_train = load_data_for_ridgeless()
coefficients = do_ridgeless_regression_and_give_back_coefficients()
coefficient_Matrix = get_coefficientMatrix_each_row_represents_a_feature()
p_value_list = get_p_value_array()
