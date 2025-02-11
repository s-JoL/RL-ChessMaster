import random
import numpy as np
from envs.gomoku_env import GomokuEnv
import multiprocessing as mp
import torch
import time

class ExperiencePool:
    def __init__(self, capacity, discard_probability_factor=0.001, board_size=15, win_length=5):
        """
        初始化经验池。去除 agent 类型相关配置，进一步精简。

        Args:
            capacity (int): 经验池的最大容量。
            discard_probability_factor (float): 抛弃概率因子。
            board_size (int): 棋盘大小.
            win_length (int): 获胜连子数.
        """
        self.capacity = capacity
        self.discard_probability_factor = discard_probability_factor
        self.global_step_count = 0
        self.board_size = board_size
        self.win_length = win_length

        self.experience_buckets = {}
        self.max_game_steps = board_size * board_size

    def add_experience(self, state, action, reward, next_state, game_step, opponent_action=None, game_id=None, creation_step=0):
        """向经验池中添加一条经验 (保持不变)."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'game_step': game_step,
            'opponent_action': opponent_action,
            'game_id': game_id,
            'creation_step': creation_step
        }
        if game_step not in self.experience_buckets:
            self.experience_buckets[game_step] = []
        bucket_capacity = self.capacity
        if len(self.experience_buckets[game_step]) >= bucket_capacity:
            index_to_remove = random.randint(0, len(self.experience_buckets[game_step]) - 1)
            self.experience_buckets[game_step].pop(index_to_remove)
        self.experience_buckets[game_step].append(experience)

    def sample_experience_batch(self, batch_size):
        """优化后的采样方法 (保持不变)."""
        non_empty_buckets = [step for step, exp in self.experience_buckets.items() if exp]
        if not non_empty_buckets:
            return []
        samples_per_bucket = max(1, batch_size // len(non_empty_buckets))
        batch = []
        for step in random.sample(non_empty_buckets, len(non_empty_buckets)):
            bucket = self.experience_buckets[step]
            batch.extend(random.sample(bucket, min(samples_per_bucket, len(bucket))))
        while len(batch) < batch_size and non_empty_buckets:
            step = random.choice(non_empty_buckets)
            if self.experience_buckets[step]:
                batch.append(random.choice(self.experience_buckets[step]))
        return random.sample(batch, min(len(batch), batch_size))

    def _generate_initial_experiences(self, num_experiences, agent_dict, global_step_count=0):
        """
        优化后的初始/周期性经验生成.  进一步简化，直接使用 agent 实例列表.

        Args:
            num_experiences (int): 生成的经验数量.
            agent_dict (dict): 智能体字典，key 为 'agent_instances' 和 'agent_probabilities'.
                                 例如: {'agent_instances': [rule_based_agent, random_agent], 'agent_probabilities': [0.8, 0.2]}.
                                 'agent_instances' 键的值应为 agent 实例列表.
            global_step_count (int): 全局步数计数器，用于记录经验创建时间.

        Returns:
            list: 生成的经验数据列表.
        """
        agent_instances = agent_dict.get('agent_instances')
        agent_probabilities = agent_dict.get('agent_probabilities')

        if not agent_instances or not agent_probabilities:
            raise ValueError("agent_dict must contain 'agent_instances' and 'agent_probabilities' keys.")
        if not np.isclose(sum(agent_probabilities), 1.0):
            raise ValueError("agent_probabilities must sum to 1.0.")

        env = GomokuEnv(board_size=self.board_size)
        experiences = []
        for game_id in range((num_experiences + 1) // 1):
            env.reset()
            game_step = 0
            while game_step < self.max_game_steps:
                state = env.board.copy()
                # 轮流使用不同类型的 agent 实例
                agent_instance = random.choices(agent_instances, weights=agent_probabilities, k=1)[0] # 直接获取 agent 实例

                action = self._select_action(env, agent_instance) #  直接传递 agent 实例
                if not action: break

                board_after_move, reward_agent, done_agent = env.step(action)

                # 对手也按相同的 agent 策略选择动作
                opponent_agent_instance = random.choices(agent_instances, weights=agent_probabilities, k=1)[0]
                opponent_action, next_state, done_opponent = self._handle_opponent_turn(env, opponent_agent_instance, board_after_move, done_agent) #  传递对手 agent 实例

                reward_value = 0
                if done_agent:
                    reward_value = 100
                elif done_opponent:
                    reward_value = -100

                experiences.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': reward_value,
                    'next_state': next_state.copy(),
                    'game_step': game_step,
                    'opponent_action': opponent_action,
                    'game_id': game_id,
                    'creation_step': global_step_count + game_step
                })

                game_step += 1
                if done_agent or done_opponent: break

        return experiences

    def _select_action(self, env, agent_instance):
        """选择动作的通用方法，直接调用 agent 实例的 select_action 方法, 去除 agent_type 参数和判断"""
        return agent_instance.select_action(env) #  直接调用 agent 实例的 select_action 方法

    def _handle_opponent_turn(self, env, opponent_agent_instance, board_after_move, done_agent):
        """处理对手回合，与 _select_action 类似，直接调用 agent 实例的 select_action 方法,  去除 opponent_agent_type 参数"""
        if done_agent:
            return None, board_after_move, True

        opponent_action = self._select_action(env, opponent_agent_instance) #  直接调用 _select_action 选择对手动作, 传递 agent 实例
        if opponent_action is None:
            return None, board_after_move, True

        next_state, reward, done_opponent = env.step(opponent_action)
        return opponent_action, next_state, done_opponent

    def update_pool_with_probabilistic_removal(self, new_experiences, current_global_step):
        """使用概率抛弃策略更新经验池 (保持不变)."""
        buckets_after_removal = {}
        removed_count = 0
        for step, bucket in self.experience_buckets.items():
            updated_bucket = []
            for experience in bucket:
                age = current_global_step - experience['creation_step']
                discard_probability = self._calculate_discard_probability(age)
                if random.random() > discard_probability:
                    updated_bucket.append(experience)
                else:
                    removed_count += 1
            if updated_bucket:
                buckets_after_removal[step] = updated_bucket
        print(f"\n经验池更新: 移除经验数: {removed_count}")
        self.experience_buckets = buckets_after_removal
        added_count = 0
        for new_experience in new_experiences:
            self.add_experience(**new_experience)
            added_count += 1
        print(f"经验池更新: 添加经验数: {added_count}")
        print(f"经验池更新: 当前经验总数: {self.get_pool_size()}")

    def _calculate_discard_probability(self, age):
        """计算基于经验年龄的抛弃概率 (保持不变)."""
        probability = self.discard_probability_factor * age
        return max(0, min(probability, 1))

    def initialize_pool(self, initial_size, agent_dict, num_processes=mp.cpu_count(), global_step_count=0):
        """
        使用多进程初始化经验池.  进一步简化, agent_dict 直接接收 agent 实例列表.

        Args:
            initial_size (int): 经验池初始填充的经验数量.
            agent_dict (dict): 智能体字典，key 为 'agent_instances' 和 'agent_probabilities'.
                                 例如: {'agent_instances': [rule_based_agent, random_agent], 'agent_probabilities': [0.8, 0.2]}.
                                 'agent_instances' 键的值应为 agent 实例列表.
            num_processes (int): 使用的进程数量，默认为 CPU 核心数.
            global_step_count (int): 全局步数计数器.
        """
        if initial_size > self.capacity:
            initial_size = self.capacity
        pool_size_per_process = initial_size // num_processes
        remainder = initial_size % num_processes

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_processes) as pool:
            results = []
            for i in range(num_processes):
                size = pool_size_per_process + (1 if i < remainder else 0)
                if size > 0:
                    results.append(pool.apply_async(
                        self._generate_initial_experiences,
                        args=(size, agent_dict, global_step_count) #  直接传递 agent_dict
                    ))
            for result in results:
                experiences = result.get()
                random.shuffle(experiences)
                for exp in experiences:
                    self.add_experience(**exp)
        total_experiences = sum(len(bucket) for bucket in self.experience_buckets.values())
        print(f"\n经验池初始化完成:")
        print(f"- 目标经验数: {initial_size}")
        print(f"- 实际生成数: {total_experiences}")
        print(f"- 步数桶数量: {len(self.experience_buckets)}")
        print(f"- 最大桶大小: {max(len(bucket) for bucket in self.experience_buckets.values())}")

    def get_pool_size(self):
        """获取当前经验池中的经验数量 (保持不变)."""
        return sum(len(bucket) for bucket in self.experience_buckets.values())

    def clear_pool(self):
        """清空经验池 (保持不变)."""
        self.experience_buckets = {}

if __name__ == '__main__':
    from agents.rule_based_agent import RuleBasedAgent
    from agents.random_agent import RandomAgent
    from agents.dqn_agent import DQNAgent
    from gui.experience_viewer import ExperienceViewer

    # 初始化 RuleBasedAgent 和 RandomAgent 实例
    rule_based_agent = RuleBasedAgent()
    random_agent = RandomAgent()
    dqn_agent = DQNAgent()

    # 定义 agent 字典，key 为 'agent_instances' 和 'agent_probabilities'
    agent_dict = {
        'agent_instances': [rule_based_agent, random_agent, dqn_agent], #  直接使用 agent 实例列表
        'agent_probabilities': [0.3, 0.3, 0.4]
    }

    experience_pool = ExperiencePool(capacity=50)
    print(f"初始化前经验池大小: {experience_pool.get_pool_size()}")
    experience_pool.initialize_pool(initial_size=100, agent_dict=agent_dict) # 传递 agent_dict
    print(f"初始化后经验池大小: {experience_pool.get_pool_size()}")

    batch = experience_pool.sample_experience_batch(batch_size=100)
    if batch:
        ExperienceViewer(batch).mainloop()
    else:
        print("经验池为空，无法采样展示。请检查经验池初始化是否成功。")
