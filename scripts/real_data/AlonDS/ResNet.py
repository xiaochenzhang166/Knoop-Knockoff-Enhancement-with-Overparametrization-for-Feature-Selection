import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error


def get_mse_from_ResNet(X_train, y_train, X_test, y_test):
    global ResidualBlock, ResNet, n_features, optimizer

    class ResidualBlock(nn.Module):
        expansion = 1  # Add 'expansion' attribute

        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += self.downsample(identity)
            out = self.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes=1):
            super(ResNet, self).__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(n_features, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self.make_layer(block, 64, layers[0], stride=1)  # Fix stride in the first layer
            self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        def make_layer(self, block, out_channels, blocks, stride=1):
            layers = []
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.in_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    # Rest of the code remains the same
    # ...
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    # 生成数据集
    y_train = y_train.astype(np.float32)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    # 转换数据形状为(n, p, 1, 1)
    X_train = X_train.view(n_samples, n_features, 1, 1)
    y_train = y_train.view(n_samples, 1)
    # 创建ResNet模型
    resnet = ResNet(ResidualBlock, [2, 2, 2, 2])  # 这里使用ResNet50的结构，可以根据需要更改深度
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = resnet(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # 测试样例
    X_test = torch.FloatTensor(X_test)
    n_samples_for_test = X_test.shape[0]
    X_test = X_test.view(n_samples_for_test, n_features, 1, 1)
    prediction = resnet(X_test).detach().numpy()
    mse = mean_squared_error(y_test, prediction)
    return mse