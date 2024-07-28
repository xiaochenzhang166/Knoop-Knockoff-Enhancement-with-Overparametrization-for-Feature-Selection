import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from parameters import data_dim, split # 参数储存在单独的文件中

# 数据读取和整理=====================================
def print_the_parameters():
    if data_dim == 2000:
        print("Every feature in Alon dataset will be loaded.")
    else:
        print(f"loading the first {data_dim} features in Alon dataset.")
    print(f"Split the dataset for test and train set at {split}.")

def load_AlonDS():
    # 读取 Excel 文件到 Pandas DataFrame
    file_path = 'AlonDS.xlsx'  # 替换为您的 Excel 文件路径
    data = pd.read_excel(file_path)
    return data


def get_X_and_y_ndarray():
    """Clean the dataset and return ndarray matrix version of features and targets"""
    X_df = data.drop(columns=['grouping']) # 获取除了 'targets' 列外的所有列作为特征列
    X = X_df.to_numpy()[:, 0:data_dim] # 获取前data_dim个features
    # 将分类问题转化为回归问题
    y = data.grouping
    y_modified = np.where(y == 'healthy', 0, y)
    y = np.where(y_modified == 'colonc', 1, y_modified)
    return X, y

def transform_targets(y_train, y_test):
    """Transform the targets with labelEncoder"""
    lab = preprocessing.LabelEncoder()
    y_train_trsf = lab.fit_transform(y_train)
    y_test_trsf = lab.fit_transform(y_test)
    return y_train_trsf, y_test_trsf

def get_train_and_test_sets():

    # Clean the dataset and return ndarray matrix version of features and targets
    X, y = get_X_and_y_ndarray()

    # split the dataset to create train and test dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

    # Transform the targets with labelEncoder
    y_train_trsf, y_test_trsf = transform_targets(y_train, y_test)

    # save the created train and test dataset,
    # including the numerical and transformed version of targets
    np.save('X_train.npy', X_train, allow_pickle=True)
    np.save('y_train.npy', y_train, allow_pickle=True)
    np.save('X_test.npy', X_test, allow_pickle=True)
    np.save('y_test.npy', y_test, allow_pickle=True)
    print("Features and numerical targets are saved.")
    np.save('y_train_trsf.npy', y_train_trsf, allow_pickle=True)
    np.save('y_test_trsf.npy', y_test_trsf, allow_pickle=True)
    print("Classification targets are saved.")

# # -----------------------------------------------
print("Preparing AlonDS for my experiment...")
print_the_parameters()
data = load_AlonDS() # load every value in AlonDS
get_train_and_test_sets()


