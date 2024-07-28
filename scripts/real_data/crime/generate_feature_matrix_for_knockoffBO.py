import numpy as np
from knockoffBO_tools import get_X_matrix_with_multiple_knockoffs
from parameters import data_dim

# ========================================================
def load_numeric_datasets():
    print("===ATTENTION: This code should only run on the REMOTE LINUX device.=== ")
    X_train = np.load('X_train.npy', allow_pickle=True)
    print(f"{data_dim} Train features are loaded.")
    return X_train

def generate_and_save_knockoff_matrix(X_train):
    # 为X_train矩阵产生多层级的knockoff
    print("Generating knockoff matrix for training features...")
    X_train_knockoff = get_X_matrix_with_multiple_knockoffs(X_train)
    np.save('X_train_knockoff.npy', X_train_knockoff)
    print("The hierachical knockoff matrix of training features is saved.")


# -------------------------------------------------------
X_train = load_numeric_datasets()
generate_and_save_knockoff_matrix(X_train)

