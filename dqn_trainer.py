"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-11 19:25:15
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-12 21:39:48
FilePath: /RL-ChessMaster/dqn_trainer.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import time
import torch
import wandb
import random
import numpy as np
import torch.optim as optim
from agents.dqn_model import DQNNet
from agents.experience_pool import ExperiencePool
from envs.gomoku_env import GomokuEnv
from agents.rule_based_agent import RuleBasedAgent # 导入 RuleBasedAgent
from agents.random_agent import RandomAgent # 导入 RandomAgent
from agents.dqn_agent import DQNAgent

class DQNTrainer:
    def __init__(self, board_size=15, learning_rate=1e-3, gamma=0.95,
                 target_update_freq=50, experience_pool_capacity=10000,
                 batch_size=2048, initial_pool_size=3000,
                 experience_pool_update_freq=100,
                 discard_probability_factor=0.0005, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化 DQN 训练器.  修改为使用 agent_dict 初始化经验池.
        """
        # Initialize wandb
        wandb.init(
            project="gomoku-dqn",
            config={
                "board_size": board_size,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "target_update_freq": target_update_freq,
                "experience_pool_capacity": experience_pool_capacity,
                "batch_size": batch_size,
                "initial_pool_size": initial_pool_size,
                "experience_pool_update_freq": experience_pool_update_freq,
                "discard_probability_factor": discard_probability_factor,
                "device": device
            }
        )
        self.device = device
        print(f"Using device: {self.device}")
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.experience_pool_capacity = experience_pool_capacity
        self.batch_size = batch_size
        self.initial_pool_size = initial_pool_size
        self.experience_pool_update_freq = experience_pool_update_freq
        self.discard_probability_factor = discard_probability_factor
        self.step_count = 0
        self.step_count_global = 0
        self.episode_count = 0

        # 初始化 Q 网络和目标网络
        self.q_net = DQNNet()
        # self.q_net.load_state_dict(torch.load('q_model_base.pth'))
        self.target_net = DQNNet()
        self.target_net.load_state_dict(self.q_net.state_dict())

        # 初始化优化器
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=learning_rate, weight_decay=1e-2)

        # 初始化经验池 (传递 discard_probability_factor, 修改为 agent_dict 初始化)
        self.experience_pool = ExperiencePool(
            capacity=experience_pool_capacity, board_size=board_size,
            discard_probability_factor=discard_probability_factor
        )

        # 初始化 Agent 实例 (用于经验池填充)
        rule_based_agent = RuleBasedAgent() #  创建 RuleBasedAgent 实例
        random_agent = RandomAgent() # 创建 RandomAgent 实例

        # 初始经验池填充策略：使用 rule_based Agent 和 Random Agent 混合策略 (修改为 agent_dict)
        agent_dict = {
            'agent_instances': [rule_based_agent, random_agent], #  初始只使用 rule_based Agent, 可以根据需要添加 RandomAgent 等
            'agent_probabilities': [0.9, 0.1] #  rule_based Agent 概率为 1.0
        }
        self.experience_pool.initialize_pool(
            initial_pool_size,
            agent_dict=agent_dict, # 传递 agent_dict
            global_step_count=self.step_count_global
        )

    def train_step(self):
        """
        执行一步 DQN 训练 (保持不变).
        """
        if self.experience_pool.get_pool_size() < self.batch_size:
            return

        self.step_count += 1
        self.step_count_global += 1
        # ending_samples = self.experience_pool.get_ending_samples()
        # batch = random.sample(ending_samples, min(self.batch_size, len(ending_samples)))
        batch = self.experience_pool.sample_experience_batch(self.batch_size)
        if not batch:
            return

        # Count terminal states in batch
        terminal_states_count = sum(1 for exp in batch if exp['is_terminated'])
        # Log the count to wandb
        wandb.log({
            "terminal_states_in_batch": terminal_states_count,
            "terminal_states_ratio": terminal_states_count / self.batch_size
        })

        batch_state = np.array([exp['state'] for exp in batch])
        batch_action = np.array([exp['action'] for exp in batch])
        batch_reward = np.array([exp['reward'] for exp in batch], dtype=np.float32)
        batch_done = np.array([exp['is_terminated'] for exp in batch], dtype=np.bool_)
        batch_next_state = np.array([exp['next_state'] if exp['next_state'] is not None else np.zeros_like(exp['state']) for exp in batch])

        state_tensor = torch.tensor(batch_state).unsqueeze(1).float().to(self.device)
        action_tensor = torch.tensor(batch_action).long().to(self.device)
        reward_tensor = torch.tensor(batch_reward).to(self.device)
        next_state_tensor = torch.tensor(batch_next_state).unsqueeze(1).float().to(self.device)
        done_mask = torch.tensor(batch_done, dtype=torch.bool).to(self.device)
        self.q_net.train()
        q_values = self.q_net(state_tensor)

        actions_index = action_tensor[:, 0] * self.board_size + action_tensor[:, 1]
        q_value = q_values.view(q_values.size(0), -1).gather(dim=1, index=actions_index.unsqueeze(1)).squeeze(1)
        
        self.target_net.eval()
        with torch.no_grad():
            next_q_values = self.target_net(next_state_tensor)
        invalid_next_state_action = torch.tensor(batch_next_state!=0).unsqueeze(1).float().to(self.device)
        next_q_values = next_q_values - invalid_next_state_action * 1e6
        max_next_q_value = next_q_values.view(next_q_values.size(0), -1).max(dim=1)[0]
        target_q_value = reward_tensor + self.gamma * max_next_q_value * (~done_mask).float()

        loss = torch.nn.MSELoss()(q_value, target_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
    
    def evaluate_ending_samples(self, threshold=0.1):
        """
        评估结束样本中Q值和真实Reward差异在阈值内的比例。
        
        Args:
            threshold (float): Q值与reward差异的可接受阈值
            
        Returns:
            float: 差异在阈值内的样本比例 (0.0 到 1.0)
        """
        ending_samples = self.experience_pool.get_ending_samples()
        if not ending_samples:
            return 0.0  # No ending samples to evaluate

        # 只评估最近的10个样本
        batch = random.sample(ending_samples, min(100, len(ending_samples)))
        
        # 转换为 batch 格式
        batch_state = np.array([exp['state'] for exp in batch])
        batch_action = np.array([exp['action'] for exp in batch])
        batch_reward = np.array([exp['reward'] for exp in batch], dtype=np.float32)

        # 转换为 tensor
        state_tensor = torch.tensor(batch_state).unsqueeze(1).float().to(self.device)
        action_tensor = torch.tensor(batch_action).long().to(self.device)
        reward_tensor = torch.tensor(batch_reward).to(self.device)
        
        self.q_net.eval()
        # 计算 Q 值
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            actions_index = action_tensor[:, 0] * self.board_size + action_tensor[:, 1]
            q_value = q_values.view(q_values.size(0), -1).gather(dim=1, index=actions_index.unsqueeze(1)).squeeze(1)

        # 计算差异在阈值内的比例
        diff = torch.abs(q_value - reward_tensor)
        within_threshold = (diff <= threshold).float().mean().item()
        
        return within_threshold
    
    def train(self, num_episodes=1000):
        """
        训练 DQN agent.  修改为使用 agent_dict 更新经验池.
        """
        print("\n开始 DQN 训练 (周期性经验池更新 + 结束样本评估)...")
        train_start_time = time.time()

        for episode in range(num_episodes):
            self.episode_count += 1
            episode_start_time = time.time()

            loss = self.train_step()

            episode_time = time.time() - episode_start_time

            print(f"Episode: {episode+1}/{num_episodes}, Avg Loss: {loss:.4f}, Time: {episode_time:.2f}s")
            # Log basic metrics every episode
            wandb.log({
                "episode": episode + 1,
                "loss": loss if loss else 0,
                "episode_time": episode_time
            })

            if (episode + 1) % 50 == 0:
                self.save_model('q_model.pth')
                # Evaluate ending samples every 50 episodes
                ending_sample_acc = self.evaluate_ending_samples()
                win_count, loss_count, draw_count = self.evaluate_agent(num_games=20)
                total_games = win_count + loss_count + draw_count
                win_rate = win_count / total_games if total_games > 0 else 0.0

                print(f"Ending Sample Acc: {ending_sample_acc:.4f}")
                print(f"--- Episode {episode+1} 评估结果: 胜: {win_count}, 负: {loss_count}, 平: {draw_count}, 胜率: {win_rate:.2f} ---")
                # Log evaluation metrics
                wandb.log({
                    "evaluation/ending_sample_acc": ending_sample_acc,
                    "evaluation/win_count": win_count,
                    "evaluation/loss_count": loss_count,
                    "evaluation/draw_count": draw_count,
                    "evaluation/win_rate": win_rate
                })
            if (episode + 1) % self.experience_pool_update_freq == 0:
                self.update_experience_pool(num_episodes)

        train_time = time.time() - train_start_time
        print(f"\nDQN 训练完成 (周期性经验池更新 + 结束样本评估)! 总耗时: {train_time:.2f}s")
        self.save_model("./dqn_model_periodic_pool_update.pth")
        print("模型已保存至 ./dqn_model_periodic_pool_update.pth")

    def update_experience_pool(self, num_episodes):
        """
        更新经验池：概率性移除旧经验，并使用动态 Agent 策略组合生成新经验.
        修改为使用 agent_dict.
        """
        print("\n--- 开始更新经验池 ---")
        update_start_time = time.time()
        num_new_experiences = 1000

        # 初始化 Agent 实例 (在 update_experience_pool 中初始化)
        rule_based_agent = RuleBasedAgent()
        random_agent = RandomAgent()

        # 动态调整 agent 策略比例 (示例)
        target_agent_probability = min(0.9, self.episode_count / num_episodes)
        rule_based_agent_probability = 1.0 - target_agent_probability - 0.1
        random_agent_probability = 0.1
        agent_probabilities = []
        agent_instances = [] #  使用 agent_instances 列表

        if target_agent_probability > 0:
            agent_instances.append(DQNAgent(q_net=self.target_net)) #  添加 target_net 实例
            agent_probabilities.append(max(0, target_agent_probability))
        if rule_based_agent_probability > 0:
            agent_instances.append(rule_based_agent) #  添加 rule_based_agent 实例
            agent_probabilities.append(max(0, rule_based_agent_probability))
        agent_instances.append(random_agent) # 添加 random_agent 实例
        agent_probabilities.append(random_agent_probability)

        # 概率归一化，确保和为 1.0
        agent_probabilities = np.array(agent_probabilities)
        agent_probabilities = agent_probabilities / np.sum(agent_probabilities)
        agent_probabilities = agent_probabilities.tolist()

        # 构建 agent_dict (使用 agent_instances 和 agent_probabilities)
        agent_dict = {
            'agent_instances': agent_instances,
            'agent_probabilities': agent_probabilities
        }

        print(f"经验池更新策略 - Agent 实例: {[agent.__class__.__name__ if not isinstance(agent, DQNNet) else 'DQNNet' for agent in agent_instances]}, 概率: {agent_probabilities}")
        # Log agent probabilities
        wandb.log({
            f"agent_prob/{agent.__class__.__name__}": prob 
            for agent, prob in zip(agent_instances, agent_probabilities)
        })
        # 使用多进程并行生成新经验  **调用 _parallel_generate_experiences**
        new_experiences = self.experience_pool._parallel_generate_experiences(num_new_experiences, agent_dict, global_step_count=self.step_count_global)
        self.experience_pool.update_pool_with_probabilistic_removal(
            new_experiences, current_global_step=self.step_count_global
        )
        update_time = time.time() - update_start_time
        print(f"--- 经验池更新完成, 耗时: {update_time:.2f}s ---")

    def evaluate_agent(self, num_games=10):
        """
        评估 agent 性能 (保持不变).
        """
        win_count = 0
        loss_count = 0
        draw_count = 0
        rule_based_agent = RuleBasedAgent(
            offense_weight=1.0,
            defense_weight=0.8
        )
        dqn_agent_eval = DQNAgent(q_net=self.q_net)

        for game_index in range(num_games):
            env = GomokuEnv(board_size=self.board_size)
            state = env.reset()
            done = False
            current_player = 1

            while not done:
                if current_player == 1:
                    action = dqn_agent_eval.select_action(env) # 修改为输入 env
                else:
                    action = rule_based_agent.select_action(env) # 修改为输入 env

                if action is None:
                    draw_count += 1
                    done = True
                    break

                next_state, reward, done = env.step(action)
                state = next_state
                current_player = env.get_current_player()

                if done:
                    if reward == 1 and current_player == 1:
                        win_count += 1
                    elif reward == 1 and current_player != 1:
                        loss_count += 1
                    break

        return win_count, loss_count, draw_count

    def save_model(self, path):
        """保存模型参数 (保持不变)."""
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        """加载模型参数 (保持不变)."""
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())

if __name__ == '__main__':
    trainer = DQNTrainer(
        board_size=15, initial_pool_size=10000, experience_pool_capacity=5000,
        experience_pool_update_freq=250, discard_probability_factor=0.005, target_update_freq=500, 
        learning_rate=1e-4, batch_size=1024
    )
    trainer.train(num_episodes=200000)
    wandb.finish()
