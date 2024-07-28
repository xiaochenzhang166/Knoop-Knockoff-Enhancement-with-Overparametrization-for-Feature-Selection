import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

def get_errorRate_from_three_layer_NN(X_train, y_train, X_test, y_test):
    # 转换为 PyTorch 的 Tensor
    x_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.int64))
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32))

    # 定义神经网络模型
    class ThreeLayerNN(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(ThreeLayerNN, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size1)
            self.layer2 = nn.Linear(hidden_size1, hidden_size2)
            self.layer3 = nn.Linear(hidden_size2, output_size)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)
            return x

    # 定义模型和优化器
    input_size = X_train.shape[1]
    hidden_size1 = 8
    hidden_size2 = 6
    num_classes = len(np.unique(y_train))  # 类别数量
    model = ThreeLayerNN(input_size, hidden_size1, hidden_size2, num_classes)  # 输出大小改为类别数量
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 使用训练好的模型进行预测
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor)
        _, y_pred = torch.max(y_pred_prob, 1)  # 获取最大概率的类别

    # 计算准确率
    errorRate = 1 - accuracy_score(y_test.astype(np.int64), y_pred.numpy())
    return errorRate
