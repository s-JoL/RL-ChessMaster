"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 20:49:59
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-09 20:50:01
FilePath: /RL-ChessMaster/envs/base_env.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import gym
import numpy as np
from gym import spaces

class BaseBoardGameEnv(gym.Env):
    """通用棋类游戏环境基类"""
    
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.done = False

        # 定义合法动作空间（所有棋盘位置）
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(board_size, board_size), dtype=np.int8
        )

    def reset(self):
        """重置棋盘"""
        self.board.fill(0)
        self.done = False
        return self.board

    def step(self, action):
        """执行一步游戏，返回新状态、奖励、是否结束"""
        raise NotImplementedError

    def render(self):
        """可视化棋盘"""
        print(self.board)

    def get_legal_actions(self):
        """返回当前合法的落子位置"""
        return list(zip(*np.where(self.board == 0)))
