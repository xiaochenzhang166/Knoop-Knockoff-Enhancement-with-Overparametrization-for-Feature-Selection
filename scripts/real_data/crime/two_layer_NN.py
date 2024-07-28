import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error




def get_mse_from_TWO_layer_NN(X_train, y_train, X_test, y_test):

    # 转换为 PyTorch 的 Tensor
    x_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32))

    # 定义神经网络模型
    class TwoLayerNN(nn.Module):
        def __init__(self, input_size, hidden_size1, output_size):
            super(TwoLayerNN, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size1)
            self.layer2 = nn.Linear(hidden_size1, output_size)  # 修改这里的变量名为 layer2

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    # 定义模型和优化器
    input_size = X_train.shape[1]
    hidden_size1 = 8
    output_size = 1
    model = TwoLayerNN(input_size, hidden_size1, output_size)
    criterion = nn.MSELoss()
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
        y_pred = model(X_test_tensor)
    # 将 Tensor 转换为 Numpy 数组
    y_pred = y_pred.numpy()
    mse = mean_squared_error(y_test, y_pred)
    return mse


