"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 21:10:04
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-09 21:10:08
FilePath: /RL-ChessMaster/agents/greedy_agent.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import numpy as np
from utils.board_utils import evaluate_board_map, calculate_position_weight, choose_best_position

class GreedyAgent:
    """
    改进后的贪心策略智能体。
    
    该 agent 先对棋盘上所有空位进行评估，生成一个评分 map，
    每个位置包含进攻分数、防守分数和二者加权的综合分数。
    在选择行动时，将综合分数乘以位置权重（靠中心得分更高），
    并选择得分最高的落子点；若无候选，则选择靠近中心的空位。
    """
    def __init__(self, board_size=15, win_length=5, offense_weight=1.0, defense_weight=1.0):
        if not isinstance(board_size, int) or board_size < 5:
            raise ValueError("board_size must be an integer >= 5")
        self.board_size = board_size
        self.win_length = win_length
        self.offense_weight = offense_weight
        self.defense_weight = defense_weight
        self.score_map = {}
        for x in range(board_size):
            for y in range(board_size):
                self.score_map[(x, y)] = {'offense': 0, 'defense': 0, 'combined': 0}

    def select_action(self, board, player):
        score_map = evaluate_board_map(board, player, self.win_length, self.offense_weight, self.defense_weight)
        best_move = None
        best_value = -float('inf')
        # 遍历所有空位，对综合分数乘以位置权重后选择最高的
        for (x, y), scores in score_map.items():
            pos_weight = calculate_position_weight(board.shape[0], x, y)
            overall = scores['combined'] * pos_weight
            if overall > best_value:
                best_value = overall
                best_move = (x, y)
        if best_move is not None:
            return best_move
        # 若无评估候选（极端情况），则选择靠中心的位置
        empty_positions = list(zip(*np.where(board == 0)))
        return choose_best_position(empty_positions, self.board_size)

    def get_evaluation_map(self, board, player):
        """
        调用之前定义的 evaluate_board_map 函数，返回一个形如：
        {(i,j): {'offense': ..., 'defense': ..., 'combined': ...}, ...}
        的评估字典。
        """
        return  evaluate_board_map(board, player, self.win_length, self.offense_weight, self.defense_weight)