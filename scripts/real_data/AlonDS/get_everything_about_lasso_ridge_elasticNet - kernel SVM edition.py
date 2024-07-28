import numpy as np
from sklearn import linear_model
import pandas as pd
from MSE_tools_with_kernel_SVM import load_datasets_for_GeneralizedLinearRegression, get_MSE_list_for_BestKFeatures
from parameters import data_dim

def get_Absolute_coefficient_for_Lars(Xtrain, ytrain):
    """
    Fit training data with lars regression,
    and get the absolute value of coefficient for each feature.
    :param Xtrain:
    :param ytrain:
    :return: absoluteValueOf_ridge_coefficient(array)
    """
    # Initialize a ridge model and fit it
    lars_model = linear_model.Lars()
    lars_model.fit(X_train, y_train)

    # Get the ridge regression coefficients
    lars_coefficients = lars_model.coef_

    # Calculate the absolute value for coefficients
    absoluteValueOf_lars_coefficient = np.abs(lars_coefficients)

    return  absoluteValueOf_lars_coefficient



def get_Absolute_coefficient_for_Ridge(Xtrain, ytrain):
    """
    Fit training data with ridge regression,
    and get the absolute value of coefficient for each feature.
    :param Xtrain:
    :param ytrain:
    :return: absoluteValueOf_ridge_coefficient(array)
    """
    # Initialize a ridge model and fit it
    ridge_model = linear_model.Ridge()
    ridge_model.fit(X_train, y_train)

    # Get the ridge regression coefficients
    ridge_coefficients = ridge_model.coef_

    # Calculate the absolute value for coefficients
    absoluteValueOf_ridge_coefficient = np.abs(ridge_coefficients)

    return  absoluteValueOf_ridge_coefficient


def get_Absolute_coefficient_for_Lasso(Xtrain, ytrain):
    """
    Fit training data with Lasso,
    and get the absolute value of coefficient for each feature.
    :param Xtrain:
    :param ytrain:
    :return: absoluteValueOf_lasso_coefficient(array)
    """
    # Initialize a lasso model and fit it
    lasso_model = linear_model.Lasso()
    lasso_model.fit(X_train, y_train)

    # Get the lasso regression coefficients
    lasso_coefficients = lasso_model.coef_

    # Calculate the absolute value for coefficients
    absoluteValueOf_lasso_coefficient = np.abs(lasso_coefficients)

    return  absoluteValueOf_lasso_coefficient



def get_Absolute_coefficient_for_elasticNet(Xtrain, ytrain):
    """
    Fit training data with elasticNet,
    and get the absolute value of coefficient for each feature.
    :param Xtrain:
    :param ytrain:
    :return: absoluteValueOf_elasticNet_coefficient(array)
    """
    # Initialize an elasticNet model and fit it
    elasticNet_model = linear_model.ElasticNet()
    elasticNet_model.fit(X_train, y_train)

    # Get the elasticNet regression coefficients
    elasticNet_coefficients = elasticNet_model.coef_

    # Calculate the absolute value for coefficients
    absoluteValueOf_elasticNet_coefficient = np.abs(elasticNet_coefficients)

    return  absoluteValueOf_elasticNet_coefficient




# ====================================================================

def find_best_K_indices_for_one_K_for_GLM(K, absoluteCoefficientsArray):
    """
    give back the best K features' indices as a list
    according to the array of absolute values of GLM coefficients.
    The larger the absolute value of the coefficient is, the better the choice will be.
    :param p-value list, list type object:
    :param K, int: 1~data_dim:
    :return: List of indices of top K largest values
    """
    if K > absoluteCoefficientsArray.shape[0]:
        print("Dimension ERROR!")

    else:
        # 找到列表中最大的K个值对应的index。如果恰好有多个值并列排在第K位，取更靠前的index。
        k_largest_values_indexArray = np.argsort(absoluteCoefficientsArray)[-1 * K:]

        # sort the indices in the list to make them ascending.
        k_largest_values_indexArray = np.sort(k_largest_values_indexArray)

        k_largest_values_indexList = k_largest_values_indexArray.tolist()

        return k_largest_values_indexList




def get_indices_from_the_best_feature_alone_to_all_features_for_GLM(absoluteCoefficientsArray):
    """
    Get a list of "best index sublists".
    The length of the sublists ranges from 1 to the number of all features.
    :return:
    """

    # initialize the list
    list_of_bestK_indices_list = []
    print(f"Finding the best 1~{data_dim} features' indices...")


    # put sublists into the list
    for K in range(1, data_dim+1):
        # 1~datadim, from only 1 feature to all features
        bestK_indices_list = find_best_K_indices_for_one_K_for_GLM(K, absoluteCoefficientsArray)
        list_of_bestK_indices_list.append(bestK_indices_list)


    return list_of_bestK_indices_list


# =======================================================
def get_MSE_list_for_LarsRegression(X_train, y_train, X_test, y_test):
    """
    Calculate MSEs for features selected by ridge regression.
    Then return the MSEs as a list.
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: MSE_list_for_RidgeRegression
    """
    # Ridge Regression
    print("For lars regression:")
    absoluteValueOf_lars_coefficients = get_Absolute_coefficient_for_Lars(X_train, y_train)
    list_of_bestK_indices_list_for_larsRegression = get_indices_from_the_best_feature_alone_to_all_features_for_GLM(absoluteValueOf_lars_coefficients)
    MSE_list_for_larsRegression = get_MSE_list_for_BestKFeatures(list_of_bestK_indices_list_for_larsRegression)

    return MSE_list_for_larsRegression




def get_MSE_list_for_RidgeRegression(X_train, y_train, X_test, y_test):
    """
    Calculate MSEs for features selected by ridge regression.
    Then return the MSEs as a list.
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: MSE_list_for_RidgeRegression
    """
    # Ridge Regression
    print("For ridge regression:")
    absoluteValueOf_ridge_coefficients = get_Absolute_coefficient_for_Ridge(X_train, y_train)
    list_of_bestK_indices_list_for_RidgeRegression = get_indices_from_the_best_feature_alone_to_all_features_for_GLM(absoluteValueOf_ridge_coefficients)
    MSE_list_for_RidgeRegression = get_MSE_list_for_BestKFeatures(list_of_bestK_indices_list_for_RidgeRegression)

    return MSE_list_for_RidgeRegression


def get_MSE_list_for_Lasso(X_train, y_train, X_test, y_test):
    """
    Calculate MSEs for features selected by lasso.
    Then return the MSEs as a list.
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: MSE_list_for_lasso
    """
    # Lasso
    print("\nFor lasso regression:")
    absoluteValueOf_lasso_coefficients = get_Absolute_coefficient_for_Lasso(X_train, y_train)
    list_of_bestK_indices_list_for_lasso = get_indices_from_the_best_feature_alone_to_all_features_for_GLM(absoluteValueOf_lasso_coefficients)
    MSE_list_for_lasso = get_MSE_list_for_BestKFeatures(list_of_bestK_indices_list_for_lasso)

    return MSE_list_for_lasso



def get_MSE_list_for_elasticNet(X_train, y_train, X_test, y_test):
    """
        Calculate MSEs for features selected by lasso.
        Then return the MSEs as a list.
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return: MSE_list_for_lasso
    """
    # ElasticNet
    print("\nFor elasticNet regression:")
    absoluteValueOf_elasticNet_coefficients = get_Absolute_coefficient_for_elasticNet(X_train, y_train)
    list_of_bestK_indices_list_for_elasticNet = get_indices_from_the_best_feature_alone_to_all_features_for_GLM(absoluteValueOf_elasticNet_coefficients)
    MSE_list_for_elasticNet = get_MSE_list_for_BestKFeatures(list_of_bestK_indices_list_for_elasticNet)

    return MSE_list_for_elasticNet

# ==========================================================
def save_everything_for_GLMs(Lars_MSE, Ridge_MSE, Lasso_MSE, ElasticNet_MSE):
    # 将所有数据存储到 DataFrame 中
    result = {'Lars': Lars_MSE,
              'Ridge': Ridge_MSE,
              'Lasso': Lasso_MSE,
              'ElasticNet': ElasticNet_MSE}
    df = pd.DataFrame(result)
    file_name = 'Result_for_GLMs - kernel SVM edition.xlsx'  # 设置输出的 Excel 文件名
    df.to_excel(file_name, index=False, sheet_name="GLMs")

# ==========================================================
X_train, y_train, X_test, y_test = load_datasets_for_GeneralizedLinearRegression()

MSE_list_for_LarsRegression = get_MSE_list_for_LarsRegression(X_train, y_train, X_test, y_test)
MSE_list_for_RidgeRegression = get_MSE_list_for_RidgeRegression(X_train, y_train, X_test, y_test)
MSE_list_for_Lasso = get_MSE_list_for_Lasso(X_train, y_train, X_test, y_test)
MSE_list_for_ElasticNet = get_MSE_list_for_elasticNet(X_train, y_train, X_test, y_test)

save_everything_for_GLMs(MSE_list_for_LarsRegression, MSE_list_for_RidgeRegression, MSE_list_for_Lasso, MSE_list_for_ElasticNet)



