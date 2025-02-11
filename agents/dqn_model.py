"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-11 19:25:15
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-11 23:23:05
FilePath: /RL-ChessMaster/agents/dqn_model.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    ResNet 风格的残差块。
    包含两个卷积层和一个 shortcut 连接。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # 批量归一化
        self.relu = nn.ReLU(inplace=True) # 原地ReLU，减少内存占用
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut 连接，如果输入通道数不等于输出通道数或者stride不为1，需要进行下采样
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """前向传播"""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut 连接
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class DQNNet(nn.Module):
    """
    全卷积 ResNet 风格的 DQN 网络，用于五子棋。
    """
    def __init__(self, num_residual_blocks=3):
        super(DQNNet, self).__init__()
        self.in_channels = 64 # 初始卷积层输出通道数

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # ResNet 残差块 layers
        self.residual_layers = self._make_residual_layers(num_residual_blocks)

        # 输出卷积层，将特征图转换为 Q 值
        self.out_conv = nn.Conv2d(self.in_channels, 1, kernel_size=1) # 输出通道为 1，表示每个位置的 Q 值

    def _make_residual_layers(self, num_blocks):
        """
        创建堆叠的残差块。
        """
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(self.in_channels, self.in_channels)) # 残差块输入输出通道数相同
        return nn.Sequential(*layers) # 使用 nn.Sequential 方便 forward 传播

    def forward(self, state):
        """前向传播"""
        # 确保输入是 float32 类型
        state = state.float()
        state = (state + 1) / 2
        # 初始卷积层
        out = self.conv1(state)
        out = self.bn1(out)
        out = self.relu(out)

        # ResNet 残差块
        out = self.residual_layers(out)

        # 输出卷积层，得到每个位置的 Q 值
        q_values = self.out_conv(out) # 输出形状为 (batch_size, 1, board_size, board_size)

        # 移除通道维度，变为 (batch_size, board_size, board_size)
        q_values = q_values.squeeze(1)
        q_values = torch.sigmoid(q_values) * 2 - 1
        return q_values

if __name__ == '__main__':
    # 单元测试
    board_size = 15
    net = DQNNet()
    # 打印网络结构
    print(net)

    # 随机生成一个状态 (模拟 batch_size=2)
    dummy_state = torch.randn(2, 1, board_size, board_size) # (batch_size, channel, height, width)

    # 前向传播
    q_values = net(dummy_state)
    print("Q Values output shape:", q_values.shape) # 应该输出 (2, board_size, board_size)
    print("Sample Q Values:", q_values[0, :3, :3]) # 打印左上角 3x3 的 Q 值
