"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-11 19:25:15
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-12 23:43:46
FilePath: /RL-ChessMaster/agents/dqn_model.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):
    def __init__(self, board_size=15, in_channels=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        参数：
          board_size: 棋盘大小，默认为15（即15x15棋盘）
          in_channels: 输入通道数。这里假设使用两个通道分别表示己方和对方的棋子状态，
                       也可以根据需要扩展，比如增加历史信息或者当前行动方标记等
        """
        super(DQNNet, self).__init__()
        self.device = device
        self.board_size = board_size
        
        # 卷积层：使用3个卷积层提取空间特征
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 卷积层输出的特征图大小为 board_size x board_size x 64，
        # 将其展平后通过全连接层映射到一个较低维度，再输出每个位置的 Q 值
        self.fc1 = nn.Linear(board_size * board_size * 64, 64)
        self.fc2 = nn.Linear(64, 512)
        self.fc3 = nn.Linear(512, board_size * board_size)
        self.to(device)
        
    def forward(self, state):
        """
        前向传播：
          输入 x 的形状为 (batch_size, in_channels, board_size, board_size)
          输出 q_values 的形状为 (batch_size, board_size * board_size)
        """
        state = state.long()
        state.to(self.device)
        # 将 1/0/-1 转换为两个通道
        my_pieces = (state == 1).float()  # 我方棋子的位置
        opponent_pieces = (state == -1).float()  # 对方棋子的位置
        state = torch.cat([my_pieces, opponent_pieces], dim=1)  # 组合成 (batch_size, 2, height, width)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 将特征图展平
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        q_values = q_values.view(q_values.shape[0], 1, self.board_size, self.board_size)
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
