import numpy as np
from pandas import read_csv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from parameters import split, data_dim

data = read_csv('energydata_complete.csv')
target = data["Appliances"]
data = data.drop(columns=['Appliances'], axis=1)


X = data.iloc[0:250, 0:data_dim].values
y = target.iloc[0:250].values


seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = seed)

# including the numerical and transformed version of targets
np.save('X_train.npy', X_train, allow_pickle=True)
np.save('y_train.npy', y_train, allow_pickle=True)
np.save('X_test.npy', X_test, allow_pickle=True)
np.save('y_test.npy', y_test, allow_pickle=True)
