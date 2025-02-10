"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 20:50:14
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-09 20:50:17
FilePath: /RL-ChessMaster/envs/gomoku_env.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
from envs.base_env import BaseBoardGameEnv
from utils.board_utils import check_winner

class GomokuEnv(BaseBoardGameEnv):
    """五子棋环境"""
    
    def __init__(self, board_size=15):
        super().__init__(board_size)
        self.current_player = 1  # 1 代表 AI，2 代表对手

    def reset(self):
        """重置棋盘"""
        self.current_player = 1
        return super().reset()

    def step(self, action):
        """执行落子"""
        x, y = action
        
        # 非法落子
        assert self.board[x, y] == 0
        
        self.board[x, y] = self.current_player
        
        # 检查是否有胜者
        if check_winner(self.board, x, y):
            self.done = True
            return self.board, 1, True  # 胜者获得奖励
        
        # 切换玩家
        # self.current_player = self.current_player % 2 + 1
        self.current_player *= -1
        return self.board, 0, False  # 继续游戏

    def get_current_player(self):
        """获取当前玩家"""
        return self.current_player
