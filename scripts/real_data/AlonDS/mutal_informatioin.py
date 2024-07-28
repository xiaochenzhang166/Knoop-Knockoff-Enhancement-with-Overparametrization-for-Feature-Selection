import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression

X_train = np.load('X_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
X_test = np.load('X_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)

MSE_list_for_mutal_information = []

for k in range(1, 11):
    fs = SelectKBest(score_func=mutual_info_regression, k=k)
    X_train_extracted = fs.fit_transform(X_train, y_train)
    X_test_extracted = fs.transform(X_test)
    linear_reg = LinearRegression()
    linear_reg.fit(X_train_extracted, y_train)
    predictions = linear_reg.predict(X_test_extracted)
    mse = mean_squared_error(y_test, predictions)
    MSE_list_for_mutal_information.append(mse)

result = {"Mutual information": MSE_list_for_mutal_information}
df = pd.DataFrame(result)
file_name = 'Result_for_Mutual information.xlsx'  # 设置输出的 Excel 文件名
df.to_excel(file_name, index=False)