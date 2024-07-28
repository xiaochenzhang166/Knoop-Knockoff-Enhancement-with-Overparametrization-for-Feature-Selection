import numpy as np
import pandas as pd
from knockpy.knockoff_filter import KnockoffFilter
from MSE_tools_with_Three_layer_NN import get_MSE_list_for_BestKFeatures, get_MSE_value_for_certain_features
from parameters import data_dim



def load_data_for_knockoff():
    """
    加载使用FS模型选择变量和计算相关变量的MSE指标时，用到的数据集
    :return:
    """
    X_train = np.load('X_train.npy', allow_pickle=True)
    y_train = np.load('y_train_trsf.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_test = np.load('y_test_trsf.npy', allow_pickle=True)

    return X_train, y_train, X_test, y_test


def randomX_Knockoff(Xtrain, ytrain):
    """
    Do model-X knockoff and return the index of the selected features
    :param Xtrain:
    :param ytrain:
    :return: index list of the selected features, W scores array
    """
    # Run model-X knockoffs
    kfilter = KnockoffFilter(
        fstat='lasso',
        ksampler='gaussian',
    )
    rejection, W_score_array = kfilter.forward(X=Xtrain, y=ytrain)
    indices_of_selected_features = []
    for index in range(rejection.shape[0]):
        if rejection[index] == 1:
            indices_of_selected_features.append(index)
    print("Number of selected features:", len(indices_of_selected_features))

    return indices_of_selected_features, W_score_array


def get_number_of_positive_W_in_W_score_array(W_scoreArray):
    """
    遍历计数，有多少变量的W statistics 取值大于0，打印此值
    :param W_scoreArray:
    :return: None
    """
    sum = 0
    for W in W_scoreArray:
        if W > 0:
            sum += 1
    print(f"There are {sum} positive W score amoung {W_scoreArray.shape}")



def find_best_K_indices_for_one_K_for_knockoff(K, W_scoreArray):
    """
    give back the best K features' indices as a list
    according to the array of W scores given by knockoff.
    The larger the scores is, the better the choice will be.
    :param p-value list, list type object:
    :param K, int: 1~data_dim:
    :return: List of indices of top K largest values
    """
    if K > W_scoreArray.shape[0]:
        print("Dimension ERROR!")

    else:
        # 找到列表中最大的K个值对应的index。如果恰好有多个值并列排在第K位，取更靠前的index。
        k_largest_values_indexArray = np.argsort(W_scoreArray)[-1 * K:]

        # sort the indices in the list to make them ascending.
        k_largest_values_indexArray = np.sort(k_largest_values_indexArray)

        k_largest_values_indexList = k_largest_values_indexArray.tolist()

        return k_largest_values_indexList




def get_indices_from_the_best_feature_alone_to_all_features_for_knockoff(W_scoreArray):
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
        bestK_indices_list = find_best_K_indices_for_one_K_for_knockoff(K, W_scoreArray)
        list_of_bestK_indices_list.append(bestK_indices_list)


    return list_of_bestK_indices_list



def save_everything_about_BestKFeatures(list_of_bestK_indices_lists, MSE_list):
    """
    Save the indices and MSEs for original knockoff.
    :return:
    """

    # 将最佳索引列表清单存入excel文件
    result = {'K': list(range(1, data_dim + 1)),
              'BestKIndicesList': list_of_bestK_indices_lists,
              'MSEList': MSE_list}

    df = pd.DataFrame(result)

    output_file = 'Result_for_Original_Knockoff - Three NN edition.xlsx'  # 设置输出的 Excel 文件名
    df.to_excel(output_file, index=False, sheet_name="Original knockoff")  # 将 DataFrame 写入 Excel，不保存索引列
    print("List of best K indices lists and mse list are saved.")



# ======================================================================
# load data
X_train, y_train, X_test, y_test = load_data_for_knockoff()
# get topK indices according to W-statistics values
indices_of_selected_features_by_ModelX, W_score_array = randomX_Knockoff(X_train, y_train)
# print how many W-statistics have positive values
get_number_of_positive_W_in_W_score_array(W_score_array)
# get the list of topk index lists
list_of_bestK_indices_list_for_knockoff = get_indices_from_the_best_feature_alone_to_all_features_for_knockoff(W_score_array)
# calculate MSEs according to the list of topK index lists
MSE_list_for_original_knockoff = get_MSE_list_for_BestKFeatures(list_of_bestK_indices_list_for_knockoff)
# save the results
save_everything_about_BestKFeatures(list_of_bestK_indices_list_for_knockoff, MSE_list_for_original_knockoff)



