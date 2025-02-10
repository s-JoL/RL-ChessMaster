"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-10 23:47:06
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-11 01:52:14
FilePath: /RL-ChessMaster/agents/experience_pool.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import random
import numpy as np
from envs.gomoku_env import GomokuEnv
from agents.greedy_agent import GreedyAgent
import multiprocessing as mp

class ExperiencePool:
    def __init__(self, capacity, board_size=15, win_length=5, offense_weight=1.0, defense_weight=1.0):
        """
        初始化经验池。

        Args:
            capacity (int): 经验池的最大容量。
            board_size (int): 棋盘大小。
            win_length (int): 获胜连子数。
            offense_weight (float): 进攻权重。
            defense_weight (float): 防守权重。
        """
        self.capacity = capacity
        self.board_size = board_size
        self.win_length = win_length
        self.offense_weight = offense_weight
        self.defense_weight = defense_weight
        
        # 按游戏步数分桶存储经验
        self.experience_buckets = {}  # {step_number: [experiences]}
        self.max_game_steps = board_size * board_size  # 最大可能步数

    def add_experience(self, state, action, reward, next_state, game_step, opponent_action=None, game_id=None):
        """
        向经验池中添加一条经验。

        Args:
            state (numpy.ndarray): 游戏状态。
            action (tuple): 执行的动作 (x, y) 坐标。
            reward (int): 获得的奖励。
            next_state (numpy.ndarray): 执行动作后的下一个游戏状态。
            game_step (int): 当前游戏进行的步数。
            opponent_action (tuple, optional): 对手的动作 (x, y) 坐标。
            game_id (int, optional): 游戏ID，用于标识同一局游戏。
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'game_step': game_step,
            'opponent_action': opponent_action,
            'game_id': game_id
        }

        # 确保桶存在
        if game_step not in self.experience_buckets:
            self.experience_buckets[game_step] = []

        # 如果当前步数的桶已满，随机移除一个经验
        bucket_capacity = self.capacity
        if len(self.experience_buckets[game_step]) >= bucket_capacity:
            index_to_remove = random.randint(0, len(self.experience_buckets[game_step]) - 1)
            self.experience_buckets[game_step].pop(index_to_remove)
        
        self.experience_buckets[game_step].append(experience)
        
    def sample_experience_batch(self, batch_size):
        """优化后的采样方法"""
        non_empty_buckets = [step for step, exp in self.experience_buckets.items() if exp]
        if not non_empty_buckets:
            return []

        samples_per_bucket = max(1, batch_size // len(non_empty_buckets))
        batch = []
        
        for step in random.sample(non_empty_buckets, len(non_empty_buckets)):
            bucket = self.experience_buckets[step]
            batch.extend(random.sample(bucket, min(samples_per_bucket, len(bucket))))

        # 补充不足的样本
        while len(batch) < batch_size and non_empty_buckets:
            step = random.choice(non_empty_buckets)
            if self.experience_buckets[step]:
                batch.append(random.choice(self.experience_buckets[step]))

        return random.sample(batch, min(len(batch), batch_size))

    def _generate_initial_experiences(self, num_experiences):
        """优化后的初始经验生成"""
        env = GomokuEnv(board_size=self.board_size)
        agent = GreedyAgent(
            board_size=self.board_size,
            win_length=self.win_length,
            offense_weight=self.offense_weight,
            defense_weight=self.defense_weight
        )

        experiences = []
        for game_id in range((num_experiences + 1) // 1):  # 每局约2步
            env.reset()
            game_step = 0
            while game_step < self.max_game_steps:
                state = env.board.copy()
                action = self._select_action(env, agent, state)
                if not action: break

                board_after_move, reward_agent, done_agent = env.step(action)
                opponent_action, next_state, done_opponent = self._handle_opponent_turn(env, agent, board_after_move, done_agent)
                
                experiences.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': 100 if done_agent else (-100 if done_opponent else 0),
                    'next_state': next_state.copy(),
                    'game_step': game_step,
                    'opponent_action': opponent_action,
                    'game_id': game_id
                })

                game_step += 1
                if done_agent or done_opponent: break

        return experiences

    def _select_action(self, env, agent, state):
        """选择动作的通用方法"""
        if random.random() < 0.1:
            empty_positions = list(zip(*np.where(state == 0)))
            return random.choice(empty_positions) if empty_positions else None
        return agent.select_action(state, env.get_current_player())

    def _handle_opponent_turn(self, env, agent, board_after_move, done_agent):
        """处理对手回合"""
        if done_agent:
            return None, board_after_move, True
            
        opponent_action = self._select_action(env, agent, board_after_move)
        if opponent_action is None:
            return None, board_after_move, True
            
        # 修改这里，确保返回值的数量与解包时一致
        next_state, reward, done_opponent = env.step(opponent_action)
        return opponent_action, next_state, done_opponent

    def initialize_pool(self, initial_size, num_processes=mp.cpu_count()):
        """
        使用多进程初始化经验池，填充多样化的初始游戏局面。

        Args:
            initial_size (int): 经验池初始填充的经验数量。
            num_processes (int): 使用的进程数量，默认为 CPU 核心数。
        """
        if initial_size > self.capacity:
            initial_size = self.capacity # 不超过容量
        pool_size_per_process = initial_size // num_processes
        remainder = initial_size % num_processes

        ctx = mp.get_context('spawn') # 避免可能的fork问题
        with ctx.Pool(processes=num_processes) as pool:
            results = []
            for i in range(num_processes):
                size = pool_size_per_process + (1 if i < remainder else 0)
                if size > 0:
                    results.append(pool.apply_async(
                        self._generate_initial_experiences,
                        args=(size,)
                    ))
            for result in results:
                experiences = result.get()
                random.shuffle(experiences)
                for exp in experiences:
                    self.add_experience(**exp) # 使用 ** 运算符传递字典参数
        total_experiences = sum(len(bucket) for bucket in self.experience_buckets.values())
        print(f"\n经验池初始化完成:")
        print(f"- 目标经验数: {initial_size}")
        print(f"- 实际生成数: {total_experiences}")
        print(f"- 步数桶数量: {len(self.experience_buckets)}")
        print(f"- 最大桶大小: {max(len(bucket) for bucket in self.experience_buckets.values())}")

    def get_pool_size(self):
        """
        获取当前经验池中的经验数量。

        Returns:
            int: 经验池中经验的数量。
        """
        return sum(len(bucket) for bucket in self.experience_buckets.values())

    def clear_pool(self):
        """
        清空经验池。
        """
        self.experience_buckets = []

if __name__ == '__main__':
    from gui.experience_viewer import ExperienceViewer  # 新增导入
    experience_pool = ExperiencePool(capacity=50)
    print(f"初始化前经验池大小: {experience_pool.get_pool_size()}")
    experience_pool.initialize_pool(initial_size=100)
    print(f"初始化后经验池大小: {experience_pool.get_pool_size()}")

    batch = experience_pool.sample_experience_batch(batch_size=100)
    if batch:
        ExperienceViewer(batch).mainloop()
    else:
        print("经验池为空，无法采样展示。请检查经验池初始化是否成功。")