"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-11 19:25:15
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-15 22:40:17
FilePath: /RL-ChessMaster/agents/base_agent.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
class BaseAgent:
    """
    五子棋智能体的基类。

    定义了所有智能体共享的基本属性和方法。
    子类需要实现 select_action 和 get_evaluation_map 方法。
    """
    def __init__(self):
        """
        初始化基类智能体。

        Args:
        """
        pass

    def select_action(self, env):
        """
        选择下一步行动的抽象方法。
        子类必须实现这个方法来定义具体的行动选择策略。

        Returns:
            tuple: 选择的落子坐标 (row, col)。
        """
        raise NotImplementedError("select_action 方法需要在子类中被实现")

    def get_evaluation_map(self, env):
        """
        获取棋盘评估 map 的抽象方法。
        子类可以实现这个方法来返回自定义的评估 map。
        默认情况下，返回一个空评分 map。

        Returns:
            dict: 评估 map，形如 {(i,j): {'offense': ..., 'defense': ..., 'combined': ...}, ...}。
        """
        raise NotImplementedError("get_evaluation_map 方法需要在子类中被实现")

    def selection_action_batch(self, envs):
        """
        批量选择下一步行动的抽象方法。
        子类必须实现这个方法来定义批量的行动选择策略。

        Returns:
            list[tuple]: 选择的落子坐标列表 [(row, col), ...]。
        """
        return [self.select_action(env) for env in envs]
