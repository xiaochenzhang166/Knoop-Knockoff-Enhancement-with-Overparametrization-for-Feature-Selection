import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from parameters import split

data = pd.read_csv("train.csv")
y = data.iloc[0:400, 81].values
X = data.iloc[0:400, 0:81].values

seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = seed)

# including the numerical and transformed version of targets
np.save('X_train.npy', X_train, allow_pickle=True)
np.save('y_train.npy', y_train, allow_pickle=True)
np.save('X_test.npy', X_test, allow_pickle=True)
np.save('y_test.npy', y_test, allow_pickle=True)