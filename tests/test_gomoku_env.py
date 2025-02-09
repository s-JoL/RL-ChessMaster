"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 21:12:10
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-09 21:12:14
FilePath: /RL-ChessMaster/tests/test_gomoku_env.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import unittest
import numpy as np
from envs.gomoku_env import GomokuEnv

class TestGomokuEnv(unittest.TestCase):
    def setUp(self):
        self.board_size = 15
        # 初始化环境（假设 BaseBoardGameEnv 在 envs/base_env.py 中已实现）
        self.env = GomokuEnv(board_size=self.board_size)
        self.env.reset()

    def test_reset(self):
        board = self.env.reset()
        # 检查返回的棋盘大小是否正确，以及棋盘全为 0
        self.assertEqual(board.shape, (self.board_size, self.board_size))
        self.assertTrue(np.all(board == 0))
        # 重置后当前玩家应为 1
        self.assertEqual(self.env.get_current_player(), 1)

    def test_valid_move(self):
        # 选择落子位置 (7,7)，对应 action = 7 * board_size + 7
        action = 7 * self.board_size + 7
        board, reward, done = self.env.step(action)
        # 落子后 (7,7) 应该为玩家 1 的棋子
        self.assertEqual(board[7, 7], 1)
        # 没有胜利也没有违规，奖励为 0，游戏未结束
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        # 落子后当前玩家应切换（此处用 1 与 -1 表示双方）
        self.assertEqual(self.env.get_current_player(), -1)

    def test_invalid_move(self):
        # 在 (7,7) 落子一次是合法的，再次在同一位置落子应被判定为非法
        action = 7 * self.board_size + 7
        _board, _reward, _done = self.env.step(action)  # 第一次合法落子
        board, reward, done = self.env.step(action)       # 重复落子
        self.assertEqual(reward, -1)
        self.assertTrue(done)

    def test_win_condition(self):
        # 构造一种胜利情形：
        # 对于玩家 1，在棋盘第 7 行连续落下四子，
        # 再在 (7,6) 落子即可形成五连珠（水平）赢得比赛。
        board = self.env.board
        board[7, 7] = 1
        board[7, 8] = 1
        board[7, 9] = 1
        board[7, 10] = 1
        # 落子在 (7,6)
        action = 7 * self.board_size + 6
        board, reward, done = self.env.step(action)
        self.assertEqual(reward, 1)
        self.assertTrue(done)

if __name__ == "__main__":
    unittest.main()
