import numpy as np
from agents.base_agent import BaseAgent  # 导入 BaseAgent 类

class RandomAgent(BaseAgent):
    def select_action(self, env): # 修改输入为 env (GomokuEnv 实例)
        """
        从所有可行的空位中随机选择一个位置作为落子动作。

        Args:
            env (GomokuEnv): GomokuEnv 实例，提供棋盘状态和游戏环境信息。

        Returns:
            tuple or None: 随机选择的落子坐标 (row, col)。
                           如果当前棋盘已满，则返回 None (虽然五子棋一般不会出现棋盘满的情况)。
        """
        legal_actions = env.get_legal_actions() # 从 env 中获取合法动作

        if not legal_actions: # 如果没有合法动作
            return None  # 返回 None 表示无法落子
        else:
            random_index = np.random.choice(len(legal_actions)) # 随机选择一个合法动作的索引
            return legal_actions[random_index] # 返回随机选择的合法动作坐标

    def get_evaluation_map(self, env):
        """
        获取棋盘评估 map。

        对于随机智能体，评估 map 实际上没有意义，因为它的行动是随机的。
        这里为了符合 BaseAgent 的接口定义，返回一个包含所有位置的评估 map，
        所有位置的 'combined' 评分都为 0。

        Args:
            env (GomokuEnv): GomokuEnv 实例，提供棋盘状态信息。

        Returns:
            dict: 包含所有位置的评估 map，所有位置的 'combined' 评分都为 0。
        """
        score_map = {}
        board_size = env.board_size # 从 env 中获取棋盘大小
        for x in range(board_size):
            for y in range(board_size):
                score_map[(x, y)] = {'combined': 0, 'offense': 0, 'defense': 0} # 所有位置的 combined 评分都为 0
        return score_map # 返回评估 map
