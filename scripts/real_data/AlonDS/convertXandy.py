import numpy as np

X = np.load("X_train.npy", allow_pickle=True)
y = np.load("y_train.npy", allow_pickle=True)
np.savetxt('X_train.csv', X, delimiter=',')
np.savetxt('y_train.csv', y, delimiter=',')