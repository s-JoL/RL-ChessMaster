"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 21:12:23
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-09 21:12:27
FilePath: /RL-ChessMaster/tests/test_greedy_agent.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import unittest
import numpy as np
from agents.greedy_agent import GreedyAgent
from envs.gomoku_env import GomokuEnv

class TestGreedyAgent(unittest.TestCase):
    def setUp(self):
        self.board_size = 15
        # 初始化环境与智能体
        self.env = GomokuEnv(board_size=self.board_size)
        self.agent = GreedyAgent(board_size=self.board_size)
        self.env.reset()

    def test_default_move_empty_board(self):
        """空棋盘时，智能体应返回靠近中心的空位（中心为 (7,7)）"""
        self.env.reset()
        board = self.env.board.copy()
        move = self.agent.select_action(board, player=1)
        expected = (self.board_size // 2, self.board_size // 2)
        self.assertEqual(move, expected, "空棋盘时应选择中心点")

    def test_greedy_agent_block_horizontal(self):
        """测试贪心智能体是否会堵对方的三连：水平三连"""
        self.env.reset()
        # 构造水平连续三个对手棋子（用 -1 表示对手）
        self.env.board[7, 7:10] = [-1] * 3
        board = self.env.board.copy()
        action = self.agent.select_action(board, 1)
        self.assertIn(action, {(7, 10), (7, 6)}, "应该堵住水平三连两端")

    def test_greedy_agent_block_vertical(self):
        """测试贪心智能体是否会堵对方的三连：垂直三连"""
        self.env.reset()
        self.env.board[7:10, 7] = [-1] * 3
        board = self.env.board.copy()
        action = self.agent.select_action(board, 1)
        self.assertIn(action, {(10, 7), (6, 7)}, "应该堵住垂直三连两端")

    def test_greedy_agent_block_positive_diagonal(self):
        """测试贪心智能体是否会堵对方的三连：正对角线三连"""
        self.env.reset()
        for i in range(3):
            self.env.board[7 + i, 7 + i] = -1
        board = self.env.board.copy()
        action = self.agent.select_action(board, 1)
        self.assertIn(action, {(10, 10), (6, 6)}, "应该堵住正对角线两端")

    def test_greedy_agent_block_negative_diagonal(self):
        """测试贪心智能体是否会堵对方的三连：反对角线三连"""
        self.env.reset()
        for i in range(3):
            self.env.board[7 + i, 9 - i] = -1
        board = self.env.board.copy()
        action = self.agent.select_action(board, 1)
        self.assertIn(action, {(10, 6), (6, 10)}, "应该堵住反对角线两端")

    def test_greedy_agent_block_edge(self):
        """测试边缘情况：只有一端可堵时"""
        self.env.reset()
        self.env.board[0, 0:3] = [-1] * 3
        board = self.env.board.copy()
        action = self.agent.select_action(board, 1)
        self.assertEqual(action, (0, 3), "边缘三连应该堵唯一可堵方向")

    def test_multiple_threats_priority(self):
        """测试多个威胁时优先处理最长连"""
        self.env.reset()
        # 在 (7,7)~(7,8) 构造对手两连，在 (5,5)~(7,5) 构造对手三连
        self.env.board[7, 7:9] = [-1] * 2  # 两连
        self.env.board[5:8, 5] = [-1] * 3  # 三连
        board = self.env.board.copy()
        action = self.agent.select_action(board, 1)
        self.assertIn(action, {(8, 5), (4, 5)}, "应该优先处理三连威胁")

    def test_greedy_agent_extend(self):
        """测试贪心智能体是否优先延续自己的棋子"""
        self.env.reset()
        # 让 AI 自己形成水平三连（己方用 1 表示）
        self.env.board[7, 7] = 1
        self.env.board[7, 8] = 1
        self.env.board[7, 9] = 1
        board = self.env.board.copy()
        action = self.agent.select_action(board, 1)
        self.assertIn(action, {(7, 10), (7, 6)}, "应该延续己方棋子形成连珠")

    def test_no_move_when_board_full(self):
        """测试当棋盘已满时，智能体应返回 None"""
        board = np.ones((self.board_size, self.board_size), dtype=int)
        action = self.agent.select_action(board, player=1)
        self.assertIsNone(action, "当棋盘已满时，返回 None")

if __name__ == "__main__":
    unittest.main()