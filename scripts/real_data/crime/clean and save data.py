import numpy as np
from pandas import read_csv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from parameters import split

attrib = read_csv('attributes.csv', delim_whitespace = True)
data = read_csv('communities.data', names = attrib['attributes'])
data = data.drop(columns=['state','county',
                          'community','communityname',
                          'fold'], axis=1)

data = data.replace('?', np.nan)
feat_miss = data.columns[data.isnull().any()]
# 选取具有缺失值的列
data_with_missing = data[feat_miss]

# 初始化 SimpleImputer，并用均值填补缺失值
imputer = SimpleImputer(strategy='mean')
data[feat_miss] = imputer.fit_transform(data_with_missing)

X = data.iloc[0:30, 0:100].values
y = data.iloc[0:30, 100].values

seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = seed)

# including the numerical and transformed version of targets
np.save('X_train.npy', X_train, allow_pickle=True)
np.save('y_train.npy', y_train, allow_pickle=True)
np.save('X_test.npy', X_test, allow_pickle=True)
np.save('y_test.npy', y_test, allow_pickle=True)
