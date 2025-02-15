"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-11 19:25:15
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-15 22:36:24
FilePath: /RL-ChessMaster/agents/rule_based_agent.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
from agents.base_agent import BaseAgent
from utils.board_utils import evaluate_board_map, calculate_position_weight

class RuleBasedAgent(BaseAgent):
    """
    改进后的贪心策略智能体。

    该 agent 先对棋盘上所有空位进行评估，生成一个评分 map，
    每个位置包含进攻分数、防守分数和二者加权的综合分数。
    在选择行动时，将综合分数乘以位置权重（靠中心得分更高），
    并选择得分最高的落子点；若无候选，则选择靠近中心的空位。
    """
    def __init__(self, offense_weight=1.0, defense_weight=0.8):
        self.offense_weight = offense_weight
        self.defense_weight = defense_weight

    def select_action(self, env): # 输入改为 env
        """选择落子动作"""
        board = env.board # 从 env 中获取棋盘
        player = env.get_current_player() # 从 env 中获取当前玩家
        score_map = evaluate_board_map(board, player, env.win_length, self.offense_weight, self.defense_weight)
        best_move = None
        best_value = -float('inf')
        # 遍历所有空位，对综合分数乘以位置权重后选择最高的
        for (x, y), scores in score_map.items():
            pos_weight = calculate_position_weight(board.shape[0], x, y)
            overall = scores['combined'] * pos_weight
            if overall > best_value:
                best_value = overall
                best_move = (x, y)
        return best_move

    def get_evaluation_map(self, env):
        """
        调用之前定义的 evaluate_board_map 函数，返回一个形如：
        {(i,j): {'offense': ..., 'defense': ..., 'combined': ...}, ...}
        的评估字典。
        """
        board = env.board # 从 env 中获取棋盘
        player = env.get_current_player() # 从 env 中获取当前玩家
        return  evaluate_board_map(board, player, env.win_length, self.offense_weight, self.defense_weight)
