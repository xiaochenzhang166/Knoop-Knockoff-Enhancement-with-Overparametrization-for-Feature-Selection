import numpy as np
from tqdm import tqdm
from knockpy.knockoff_filter import KnockoffFilter
from parameters import knockoff_hier

########################################
# The supportive tools for knockoffBO  #
########################################


def create_knockoff_once(X):
    """Get a knockoff matrix of matrix X.
    y should be classification type targets"""

    kfilter = KnockoffFilter(
        fstat='ridge',
        ksampler='gaussian',
    ) # 可以回过头来思考这些参数是否更改
    rejections = kfilter.forward(X=X, y=np.zeros(X.shape[0])) # y是否可以使用任意值？
    X = np.hstack((X, kfilter.Xk))
    return X


def normalize_rows(matrix):
    """
    归一化矩阵的每一行，使其L2范数为1。
    参数：    matrix (numpy.ndarray)：输入的矩阵。
    返回：    normalized_matrix (numpy.ndarray)：每一行都被归一化为L2范数为1的矩阵。
    """

    # 计算每一行的L2范数
    row_norms = np.linalg.norm(matrix, axis=1)
    # 归一化每一行，使其L2范数为1
    normalized_matrix = matrix / row_norms[:, np.newaxis]
    return normalized_matrix


def get_X_matrix_with_multiple_knockoffs(X, seed=42):

    np.random.seed(seed)  # Set the random seed

    # create hierachical knockoff for knockoff_hier times
    for i in tqdm(range(knockoff_hier)):
        X = create_knockoff_once(X)

    # normalize matrix by row (every l2 norm of a row is 1)
    normalized_X = normalize_rows(X)

    return normalized_X

