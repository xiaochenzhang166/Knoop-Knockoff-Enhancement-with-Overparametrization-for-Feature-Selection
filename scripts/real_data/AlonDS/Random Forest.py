import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from MSE_tools import get_MSE_value_for_certain_features

X_train = np.load('X_train.npy', allow_pickle=True)
y_train_transformed = np.load('y_train_trsf.npy', allow_pickle=True)


sel = SelectFromModel(RandomForestClassifier(n_estimators = 50))
sel.fit(X_train, y_train_transformed)
whether_the_index_is_selected_or_not = sel.get_support()

indices_of_selected_features = []
for index in range(whether_the_index_is_selected_or_not.shape[0]):
    if whether_the_index_is_selected_or_not[index] == True:
        indices_of_selected_features.append(index)

index_array_of_selected_features = np.array(indices_of_selected_features)
print("How many features are selected:", index_array_of_selected_features.shape[0])
MSE = get_MSE_value_for_certain_features(index_array_of_selected_features)
print(MSE)


# result = {"Forward Selection": MSE_list_for_forward_selection}
# df = pd.DataFrame(result)
# file_name = 'Result_for_Forward_Selection.xlsx'  # 设置输出的 Excel 文件名
# df.to_excel(file_name, index=False)



