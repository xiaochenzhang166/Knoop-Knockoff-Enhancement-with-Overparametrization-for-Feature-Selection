import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# including the numerical and transformed version of targets
X_train = np.load('X_train.npy',  allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
X_test = np.load('X_test.npy',  allow_pickle=True)
y_test = np.load('y_test.npy',  allow_pickle=True)


y_train_try_to_cut = pd.qcut(y_train, q=8)
print(y_train_try_to_cut[0:20])
bins = [0,21,30,45,50,79,88,92, 94,130]
y_train_cut = pd.cut(y_train, bins)
y_test_cut = pd.cut(y_test, bins)
print(y_train_cut)
print(y_test_cut)

label_encoder = LabelEncoder()
y_train_trsf = label_encoder.fit_transform(y_train_cut)
y_test_trsf = label_encoder.fit_transform(y_test_cut)
print(y_train_trsf)
print(y_test_trsf)

np.save("y_train_trsf.npy", y_train_trsf, allow_pickle=True)
np.save("y_test_trsf.npy", y_test_trsf, allow_pickle=True)

