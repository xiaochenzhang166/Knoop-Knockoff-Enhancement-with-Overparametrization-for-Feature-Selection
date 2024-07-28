import numpy as np

# 加载数据
y_train = np.load('y_train.npy',allow_pickle=True)
y_test = np.load('y_test.npy',allow_pickle=True)
# 转换数据类型为整数
y_train = y_train.astype(int)
y_test = y_test.astype(int)
# 统计每个类别的样本数
train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)

# 输出结果
print("在 y_train 中:")
print("类别 0 的样本数:", train_counts[0])
print("类别 1 的样本数:", train_counts[1])

print("\n在 y_test 中:")
print("类别 0 的样本数:", test_counts[0])
print("类别 1 的样本数:", test_counts[1])
