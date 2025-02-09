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
from utils.board_utils import find_best_block, find_best_extend, choose_best_position

class GreedyAgent:
    """贪心策略智能体

    策略说明：
      1. 如果对手存在明显威胁（例如连续棋型），则优先选择堵截对手。
      2. 否则，选择扩展己方棋型的最佳位置。
      3. 如果以上策略都没有生成合适位置，则选择靠近棋盘中心的空位。
    """
    
    def __init__(self, board_size=15):
        if not isinstance(board_size, int) or board_size < 5:
            raise ValueError("board_size must be an integer >= 5")
        self.board_size = board_size

    def select_action(self, board, player):
        """
        选择落子位置。

        策略步骤：
          1. 判断对手棋子（如果玩家用 1 或 -1，则对手为 -player；否则假设玩家为 1 与 2）。
          2. 调用 find_best_block 尝试堵截对手形成威胁的棋型。
          3. 如果无堵截威胁，则调用 find_best_extend 扩展己方棋型。
          4. 若前述策略都无法生成候选位置，则从所有空位中选择一个离中心最近的点。

        Args:
            board (np.ndarray): 当前棋盘状态，形状为 (board_size, board_size)，空位为 0，其它数值代表棋子。
            player (int): 当前玩家标识（例如 1 或 2，或 1 或 -1）。

        Returns:
            tuple or None: 最佳落子位置 (x, y)；如果棋盘已满，则返回 None。
        """
        # 计算对手棋子标识
        if player in [1, -1]:
            opponent = -player
        else:
            opponent = 2 if player == 1 else 1

        # 1. 尝试堵截对手威胁
        block_move = find_best_block(board, opponent)
        if block_move is not None:
            return block_move

        # 2. 尝试扩展己方棋型
        extend_move = find_best_extend(board, player, win_length=5)
        if extend_move is not None:
            return extend_move

        # 3. 默认策略：若以上均未生成落点，则从所有空位中选择离中心最近的点
        empty_positions = list(zip(*np.where(board == 0)))
        return choose_best_position(empty_positions, self.board_size)


# ===== 示例：如何使用 GreedyAgent =====

if __name__ == "__main__":
    # 创建一个空棋盘
    board_size = 15
    board = np.zeros((board_size, board_size), dtype=int)
    
    # 模拟一些落子（例如，己方与对手）
    # 假设玩家 1 为己方，玩家 2 为对手（或者使用 1 与 -1 表示）
    board[7, 7] = 1
    board[7, 8] = 1
    board[6, 7] = 2
    board[8, 7] = 2

    # 创建智能体，并选择下一步落子位置
    agent = GreedyAgent(board_size=board_size)
    action = agent.select_action(board, player=1)
    print("GreedyAgent 建议的落子位置：", action)
