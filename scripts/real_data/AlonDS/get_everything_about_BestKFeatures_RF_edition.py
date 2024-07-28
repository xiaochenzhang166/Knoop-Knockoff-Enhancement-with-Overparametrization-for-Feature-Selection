import numpy as np
import pandas as pd
import pickle
from MSE_tools_with_RF import get_MSE_list_for_BestKFeatures, get_bestKfeatures_indices_as_list
from parameters import data_dim



def load_pValues_as_a_list():
    """
    读取所需的p-value数据，返回一个list
    :return:
    """
    p_value_array = np.load("p-values.npy")
    p_value_list = p_value_array.tolist()
    return p_value_list



def find_best_K_indices_for_one_K(K, pValueList):
    """
    give back the best K features' indices as a list according to the p-value array.
    The smaller the p-value is, the better the choice will be.
    :param p-value list, list type object:
    :param K, int: 1~data_dim:
    :return: List of indices of top K largest values
    """

    # 找到列表中最小的K个值对应的index。如果恰好有多个值并列排在第K位，取更靠前的index。
    k_smallest_values_indexArray = np.argsort(pValueList)[0:K]

    # sort the indices in the list to make them ascending.
    k_smallest_values_indexArray = np.sort(k_smallest_values_indexArray)

    k_smallest_values_indexList = k_smallest_values_indexArray.tolist()

    return k_smallest_values_indexList



def get_indices_from_the_best_feature_alone_to_all_features(p_value_list):
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
        bestK_indices_list = find_best_K_indices_for_one_K(K, p_value_list)
        list_of_bestK_indices_list.append(bestK_indices_list)


    return list_of_bestK_indices_list



def find_the_worst_pValue_for_one_BestKIndices(K, p_value_list, list_of_bestK_indices_lists):
    """
    Find the Kth smallest p-value.
    :param K:
    :return:
    """
    # get the best K features' indices
    bestKIndices = get_bestKfeatures_indices_as_list(K, list_of_bestK_indices_lists)

    # get the worst p-value
    p_value_array = np.array(p_value_list)
    BestK_pValues_are_no_larger_than = max(p_value_array[bestKIndices])

    return BestK_pValues_are_no_larger_than



def get_the_worst_pValues_for_all_BestKIndices(p_value_list, list_of_bestK_indices_lists):
    """
    Get the worst p-value of the best features.
    :return:
    """
    print("Finding the largest p-values for best features... ")
    list_of_worst_pValues = []
    for K in range(1, data_dim+1):
        worst_pValue = find_the_worst_pValue_for_one_BestKIndices(K, p_value_list, list_of_bestK_indices_lists)
        list_of_worst_pValues.append(worst_pValue)

    return list_of_worst_pValues




def save_everything_about_BestKFeatures(list_of_bestK_indices_lists, list_of_worst_pValues, MSE_list):
    """
    Save the indices, worst p-values and MSEs.
    :return:
    """

    # 将最佳索引列表清单存入excel文件
    result = {'K': list(range(1, data_dim + 1)),
              'BestKIndicesList': list_of_bestK_indices_lists,
              'WorstpValue': list_of_worst_pValues,
              'MSEList': MSE_list}

    df = pd.DataFrame(result)

    output_file = 'Result_for_KnockoffBO-RF edition.xlsx'  # 设置输出的 Excel 文件名
    df.to_excel(output_file, index=False, sheet_name="knockoffBO")  # 将 DataFrame 写入 Excel，不保存索引列
    print("List of best K indices lists, the worst p-values and mse list are saved as BestKFeatures.xlsx.")


def load_list_of_bestK_indices_lists_for_knockoffBO():
    """
    如何读取储存的最佳索引:
    从文件中加载knockoffBO的最佳变量索引清单，然后返回
    :return:索引清单list_of_bestK_indices_lists
    """
    with open('list_of_bestK_indices_lists.pkl', 'rb') as file:
        list_of_bestK_indices_lists = pickle.load(file)
    return list_of_bestK_indices_lists


# ================================================




def main():
    # 载入数据
    p_value_list = load_pValues_as_a_list()

    # 得到最佳索引列表清单
    list_of_bestK_indices_lists = get_indices_from_the_best_feature_alone_to_all_features(p_value_list)

    # 得到第K个最佳变量对应的p-value清单
    list_of_worst_pValues = get_the_worst_pValues_for_all_BestKIndices(p_value_list, list_of_bestK_indices_lists)

    # 得到第K个最佳变量对应的MSE清单
    MSE_list = get_MSE_list_for_BestKFeatures(list_of_bestK_indices_lists)

    # 组合各清单并存储
    save_everything_about_BestKFeatures(list_of_bestK_indices_lists, list_of_worst_pValues, MSE_list)





if __name__ == "__main__":
    main()