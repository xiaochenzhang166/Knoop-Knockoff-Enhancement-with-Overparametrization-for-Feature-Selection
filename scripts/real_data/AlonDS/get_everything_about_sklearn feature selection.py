from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector


X_train = np.load('X_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
X_test = np.load('X_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)



# forward selection==================================================
MSE_list_for_forward_selection = []
for num_fea in range(1, 11):
    linear = linear_model.LinearRegression()
    sfs = SequentialFeatureSelector(linear, n_features_to_select=num_fea,cv=2,direction='forward')
    sfs.fit(X_train, y_train)
    X_train_top_K = sfs.transform(X_train)
    X_test_top_K =sfs.transform(X_test)
    # 对这些变量进行线性回归
    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(X_train_top_K, y_train)
    predictions = linear_reg.predict(X_test_top_K)
    forward_mse = mean_squared_error(y_test, predictions)
    MSE_list_for_forward_selection.append(forward_mse)

result = {"Forward Selection": MSE_list_for_forward_selection}
df = pd.DataFrame(result)
file_name = 'Result_for_Forward_Selection.xlsx'  # 设置输出的 Excel 文件名
df.to_excel(file_name, index=False)



