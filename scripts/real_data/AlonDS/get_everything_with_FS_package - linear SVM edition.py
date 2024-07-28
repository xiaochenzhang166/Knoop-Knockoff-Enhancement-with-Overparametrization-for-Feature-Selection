import numpy as np
import pandas as pd
from MSE_tools_with_linear_SVM import get_MSE_value_for_certain_features
from FS.ba import jfs as ba_jfs
from FS.cs import jfs as cs_jfs
from FS.de import jfs as de_jfs
from FS.fa import jfs as fa_jfs
from FS.fpa import jfs as fpa_jfs
from FS.ga import jfs as ga_jfs
from FS.gwo import jfs as gwo_jfs
from FS.hho import jfs as hho_jfs
from FS.ja import jfs as ja_jfs
from FS.pso import jfs as pso_jfs
from FS.sca import jfs as sca_jfs
from FS.ssa import jfs as ssa_jfs
from FS.woa import jfs as woa_jfs
# change this to switch algorithm
# ba cs de fa fpa ga gwo hho ja pso sca ssa woa
from sklearn.model_selection import train_test_split


# ======================================================
def load_data_for_FS_package():
    """
    加载使用FS模型选择变量和计算相关变量的MSE指标时，用到的数据集
    :return:
    """
    X_train = np.load('X_train.npy', allow_pickle=True)
    y_train = np.load('y_train_trsf.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_test = np.load('y_test_trsf.npy', allow_pickle=True)

    return X_train, y_train, X_test, y_test

# ============================================================
def get_jfs_model(jfs, X_train, y_train, X_test, y_test):
    """
    在传入的数据集上，运行传入的jfs模型
    :param jfs:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:运行完成的jfs模型
    """
    feat = X_train
    label = y_train
    fold = {'xt': X_train, 'yt': y_train, 'xv': X_test, 'yv': y_test}
    # parameter
    k = 5  # k-value in KNN
    N = 10  # number of chromosomes
    T = 100  # maximum number of generations
    opts = {'k': k, 'fold': fold, 'N': N, 'T': T}
    # perform feature selection
    fmdl = jfs(feat, label, opts)
    return fmdl

# ==========================================================
def get_MSE_for_BA(X_train, y_train, X_test, y_test):
    """
    使用BA模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(ba_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE


def get_MSE_for_CS(X_train, y_train, X_test, y_test):
    """
    使用CS模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(cs_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE

def get_MSE_for_DE(X_train, y_train, X_test, y_test):
    """
    使用DE模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(de_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE


def get_MSE_for_FA(X_train, y_train, X_test, y_test):
    """
    使用FA模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(fa_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE

def get_MSE_for_FPA(X_train, y_train, X_test, y_test):
    """
    使用FPA模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(fpa_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE


def get_MSE_for_GA(X_train, y_train, X_test, y_test):
    """
    使用GA模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(ga_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE

def get_MSE_for_GWO(X_train, y_train, X_test, y_test):
    """
    使用GWO模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(gwo_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE

def get_MSE_for_HHO(X_train, y_train, X_test, y_test):
    """
    使用HHO模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(hho_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE


def get_MSE_for_JA(X_train, y_train, X_test, y_test):
    """
    使用JA模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(ja_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE


def get_MSE_for_PSO(X_train, y_train, X_test, y_test):
    """
    使用PSO模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(pso_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE


def get_MSE_for_SCA(X_train, y_train, X_test, y_test):
    """
    使用SCA模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(sca_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE


def get_MSE_for_SSA(X_train, y_train, X_test, y_test):
    """
    使用SSA模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(ssa_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE


def get_MSE_for_WOA(X_train, y_train, X_test, y_test):
    """
    使用WOA模型进行变量选择，返回选中的变量数和相关变量的MSE
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:num_feat, MSE
    """
    fmdl = get_jfs_model(woa_jfs, X_train, y_train, X_test, y_test)
    # array of selected features' indices
    sf = fmdl['sf']
    # number of selected features
    num_feat = fmdl['nf']
    MSE = get_MSE_value_for_certain_features(sf)
    return num_feat, MSE

# ===================================================================================
def main():
    X_train, y_train, X_test, y_test = load_data_for_FS_package()
    X_train_for_FS, X_validation_for_FS, y_train_for_FS, y_validation_for_FS = train_test_split(X_train, y_train,
                                                                                                test_size=0.2,
                                                                                                random_state=123)
    hho_num_fea, hho_MSE = get_MSE_for_HHO(X_train_for_FS, y_train_for_FS, X_validation_for_FS, y_validation_for_FS)
    ja_num_fea, ja_MSE = get_MSE_for_JA(X_train_for_FS, y_train_for_FS, X_validation_for_FS, y_validation_for_FS)
    sca_num_fea, sca_MSE = get_MSE_for_SCA(X_train_for_FS, y_train_for_FS, X_validation_for_FS, y_validation_for_FS)
    ssa_num_fea, ssa_MSE = get_MSE_for_SSA(X_train_for_FS, y_train_for_FS, X_validation_for_FS, y_validation_for_FS)
    woa_num_fea, woa_MSE = get_MSE_for_WOA(X_train_for_FS, y_train_for_FS, X_validation_for_FS, y_validation_for_FS)
    FS_method_name_list = ["hho", "ja", "sca", "ssa", "woa"]
    num_fea_list = [hho_num_fea, ja_num_fea, sca_num_fea, ssa_num_fea, woa_num_fea]
    MSE_list = [hho_MSE, ja_MSE, sca_MSE, ssa_MSE, woa_MSE]
    result = {'Name': FS_method_name_list,
              'featureNumbers': num_fea_list,
              'errorRate': MSE_list}
    df = pd.DataFrame(result)
    file_name = 'Result_for_FS- linear SVM edition.xlsx'  # 设置输出的 Excel 文件名
    df.to_excel(file_name, index=False, sheet_name="FS")


# ========================================================
if __name__ == "__main__":
    main()