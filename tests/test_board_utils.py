"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 21:11:53
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-09 21:11:59
FilePath: /RL-ChessMaster/tests/test_board_utils.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import unittest
import numpy as np

from utils.board_utils import (check_winner, is_alive_pattern, analyze_direction, evaluate_position,
                    choose_best_position, calculate_position_weight, find_best_block, find_best_extend)

class TestGomokuFunctions(unittest.TestCase):
    def setUp(self):
        """初始化测试环境，创建各种测试棋型"""
        self.board_size = 15
        # 注意：这里明确指定 dtype=int，确保棋盘中存储整数（1、-1、2等）
        self.empty_board = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # 水平连五：在第7行，列5到9连续放置 1
        self.horizontal_five = self.empty_board.copy()
        self.horizontal_five[7, 5:10] = 1
        
        # 垂直连四：在第7列，行4到7连续放置 1
        self.vertical_four = self.empty_board.copy()
        self.vertical_four[4:8, 7] = 1
        
        # 对角线连三：从 (5,5) 开始，连续放置 3 个 1
        self.diagonal_three = self.empty_board.copy()
        for i in range(3):
            self.diagonal_three[5 + i, 5 + i] = 1
            
        # 活三：在第7行，列6到8放置 1，且两端为空
        self.alive_three = self.empty_board.copy()
        self.alive_three[7, 6:9] = 1
        
        # 死三：在第7行，列6到8放置 1，但在右侧（列9）放置 -1，模拟一端被封堵
        self.dead_three = self.empty_board.copy()
        self.dead_three[7, 6:9] = 1
        self.dead_three[7, 9] = -1
        
        # 跳三：在第7行，放置 1 在列6、7和9，中间跳过列8
        self.jump_three = self.empty_board.copy()
        self.jump_three[7, [6, 7, 9]] = 1

    # ---------- 测试 check_winner ----------
    def test_check_winner_horizontal_five(self):
        """测试水平连五时，check_winner 应返回 True"""
        # 选择中间位置进行检查
        result = check_winner(self.horizontal_five, 7, 7, win_length=5)
        self.assertTrue(result)

    def test_check_winner_vertical_four(self):
        """测试垂直连四时（棋子数不足 5），check_winner 返回 False；但当 win_length=4 时返回 True"""
        result = check_winner(self.vertical_four, 6, 7, win_length=5)
        self.assertFalse(result)
        result = check_winner(self.vertical_four, 6, 7, win_length=4)
        self.assertTrue(result)

    def test_check_winner_diagonal_three(self):
        """测试对角线连三：win_length=3 应返回 True，win_length=5 返回 False"""
        result = check_winner(self.diagonal_three, 6, 6, win_length=3)
        self.assertTrue(result)
        result = check_winner(self.diagonal_three, 6, 6, win_length=5)
        self.assertFalse(result)

    def test_check_winner_no_piece(self):
        """测试空棋盘的某个位置无棋子时返回 False"""
        result = check_winner(self.empty_board.copy(), 7, 7, win_length=5)
        self.assertFalse(result)

    def test_check_winner_small_board_exception(self):
        """测试棋盘尺寸小于 win_length 时应抛出异常"""
        small_board = np.zeros((3, 3), dtype=int)
        small_board[1, 1] = 1
        with self.assertRaises(ValueError):
            check_winner(small_board, 1, 1, win_length=5)

    # ---------- 测试 is_alive_pattern ----------
    def test_is_alive_pattern_alive_three(self):
        """测试活三：应返回 True（两端为空）"""
        result = is_alive_pattern(self.alive_three, 7, 7, length=3)
        self.assertTrue(result)

    def test_is_alive_pattern_dead_three(self):
        """测试死三：一端被封堵，返回 False"""
        result = is_alive_pattern(self.dead_three, 7, 7, length=3)
        self.assertFalse(result)

    def test_is_alive_pattern_not_enough(self):
        """测试棋子数不足时返回 False"""
        board = self.empty_board.copy()
        board[7, 7] = 1
        board[7, 8] = 1
        result = is_alive_pattern(board, 7, 7, length=3)
        self.assertFalse(result)

    # ---------- 测试 analyze_direction ----------
    def test_analyze_direction_continuous(self):
        """测试 analyze_direction：在无跳子情况下的连续子计算"""
        board = self.empty_board.copy()
        board[5, 5] = 1
        board[5, 6] = 1  # 向右连续
        directions = [(0, 1), (0, -1)]
        pure, bonus, blocks, jump_used = analyze_direction(board, 5, 5, 1, directions)
        # 预期：pure_count = 1（当前位置）+1（右侧连续）=2；无跳子 bonus；无封堵；jump_used 为 False
        self.assertEqual(pure, 2)
        self.assertEqual(bonus, 0)
        self.assertEqual(blocks, 0)
        self.assertFalse(jump_used)

    def test_analyze_direction_jump(self):
        """测试 analyze_direction 检测跳子情况"""
        # 使用 jump_three 棋盘，在第7行，中心位置 (7,7) 应检测到右侧跳子（因为 (7,8) 空，而 (7,9)==1）
        directions = [(0, 1), (0, -1)]
        pure, bonus, blocks, jump_used = analyze_direction(self.jump_three, 7, 7, 1, directions)
        # 预期：
        # 向左：从 (7,7) 看，(7,6)==1 连续，因此连续数 +1；
        # 向右：(7,8)==0，但 (7,9)==1，检测到跳子
        # pure_count = 1（中心）+1（向左）=2，bonus 应为 1，因为右侧跳子且左侧已有连续子（根据实现，可设为 1 或 0.5，此处测试依据前述代码逻辑）
        self.assertEqual(pure, 2)
        # 根据之前的代码逻辑：如果一侧有连续子，则跳子的 bonus 为 0.5；但如果两侧都检测到跳子 bonus 为 1。
        # 这里只有右侧检测到跳子，所以预期 bonus 为 1 或 0.5，具体值视实现而定。
        # 若使用上面提供的实现逻辑，则：
        self.assertIn(bonus, [0.5, 1])
        self.assertTrue(jump_used)

    # ---------- 测试 evaluate_position ----------
    def test_evaluate_position_empty_board(self):
        """测试空棋盘上评估落子位置返回的分值、最大连子数和方向类型"""
        score, effective_count, best_direction = evaluate_position(self.empty_board.copy(), 7, 7, 1, win_length=5)
        self.assertIsInstance(score, int)
        self.assertGreaterEqual(effective_count, 1)
        self.assertIn(best_direction, [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ])

    def test_evaluate_position_alive_three(self):
        """测试活三局面下评估函数的返回值"""
        # 以 alive_three 棋盘为例，选择 (7,5) 作为候选位置进行扩展
        score, effective_count, best_direction = evaluate_position(self.alive_three.copy(), 7, 5, 1, win_length=5)
        self.assertIsInstance(score, int)
        self.assertGreater(effective_count, 1)
        self.assertIsNotNone(best_direction)

    # ---------- 测试 choose_best_position ----------
    def test_choose_best_position_normal(self):
        """测试从多个候选位置中选择最靠近中心的点"""
        positions = [(0, 0), (7, 7), (14, 14)]
        best = choose_best_position(positions, self.board_size)
        self.assertEqual(best, (7, 7))

    def test_choose_best_position_invalid_type(self):
        """测试传入错误类型的坐标时抛出 TypeError"""
        positions = [("a", 0)]
        with self.assertRaises(TypeError):
            choose_best_position(positions, self.board_size)

    def test_choose_best_position_out_of_bounds(self):
        """测试传入超出棋盘范围的坐标时抛出 ValueError"""
        positions = [(16, 0)]
        with self.assertRaises(ValueError):
            choose_best_position(positions, self.board_size)

    def test_choose_best_position_empty(self):
        """测试候选位置为空时返回 None"""
        best = choose_best_position([], self.board_size)
        self.assertIsNone(best)

    # ---------- 测试 calculate_position_weight ----------
    def test_calculate_position_weight_center_and_corner(self):
        """测试中心位置权重应为 1，角落位置权重应较低"""
        weight_center = calculate_position_weight(self.board_size, 7, 7)
        weight_corner = calculate_position_weight(self.board_size, 0, 0)
        self.assertAlmostEqual(weight_center, 1)
        self.assertAlmostEqual(weight_corner, 0.5)

    # ---------- 测试 find_best_block ----------
    def test_find_best_block_detects_threat(self):
        """测试 find_best_block 能够检测到对手威胁并返回堵截位置"""
        board = self.empty_board.copy()
        opponent = 2
        # 构造对手威胁：在第7行，列5到8放置 opponent 棋子（连续 4 子，威胁形成连五）
        board[7, 5:9] = opponent
        best_block = find_best_block(board, opponent)
        # 根据实现，最佳堵截点可能为 (7,4) 或 (7,9)
        self.assertIn(best_block, [(7, 4), (7, 9)])

    def test_find_best_block_no_threat(self):
        """测试当无对手威胁时，find_best_block 返回 None"""
        board = self.empty_board.copy()
        opponent = 2
        best_block = find_best_block(board, opponent)
        self.assertIsNone(best_block)

    # ---------- 测试 find_best_extend ----------
    def test_find_best_extend_normal(self):
        """测试 find_best_extend 在存在扩展潜力时返回合理落子点"""
        board = self.empty_board.copy()
        player = 1
        # 构造局面：在第7行放置一个棋子
        board[7, 7] = player
        best_extend = find_best_extend(board, player, win_length=5)
        self.assertIsNotNone(best_extend)
        self.assertTrue(0 <= best_extend[0] < self.board_size)
        self.assertTrue(0 <= best_extend[1] < self.board_size)

    def test_find_best_extend_full_board(self):
        """测试满棋盘情况下，find_best_extend 返回 None"""
        board = np.ones((self.board_size, self.board_size), dtype=int)
        player = 1
        best_extend = find_best_extend(board, player, win_length=5)
        self.assertIsNone(best_extend)

if __name__ == '__main__':
    unittest.main()
