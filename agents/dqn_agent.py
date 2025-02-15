"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-11 19:25:15
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-15 22:50:40
FilePath: /RL-ChessMaster/agents/dqn_agent.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import torch
from agents.base_agent import BaseAgent  # 确保导入了 BaseAgent，即使 DQN Agent 可能不直接继承它
from agents.dqn_model import DQNNet  # 假设 DQNNet 定义在 agents.dqn_model.py 中

class DQNAgent(BaseAgent):
    """
    基于 DQN 的智能体。

    该 agent 使用训练好的 DQN 模型评估棋盘状态，
    并选择 Q 值最大的合法动作。
    与 RuleBasedAgent 保持接口一致，但内部使用 DQN 网络进行决策。
    """
    def __init__(self, model_path=None, q_net=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化 DQN Agent。

        Args:
            model_path (str): 训练好的 DQN 模型文件路径 (当 q_net=None 时使用).
            q_net (DQNNet, optional):  **直接传入的 DQN 模型实例。 如果提供，则忽略 model_path 加载。** 默认为 None.
        """
        self.device = device
        self.model_path = model_path
        self._q_net_instance = q_net #  保存传入的 q_net 实例

        if q_net is not None: # **如果 q_net 参数被传入，则直接使用传入的模型**
            print("DQNAgent 初始化: 使用直接传入的 DQN 模型实例.")
            self.q_net = q_net #  使用传入的模型
            self.q_net.eval() # 设置为评估模式
        elif model_path is not None: # **否则，仍然从 model_path 加载模型**
            self.q_net = DQNNet() #  初始化 DQN 网络 (如果 q_net 没有传入)
            self._load_model() # 加载预训练模型
        else:
            self.q_net = DQNNet() #  初始化 DQN 网络 (如果 q_net 没有传入)
            self.q_net.eval() # 设置为评估模式

    def _load_model(self):
        """加载训练好的 DQN 模型参数 (仅当 q_net 未直接传入时使用)。"""
        if self._q_net_instance is not None: # 如果已经有传入的 q_net 实例，则跳过文件加载
            print("跳过模型文件加载，因为已提供 q_net 实例。")
            return #  直接返回，不加载文件

        try:
            self.q_net.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'))) # 加载模型到 CPU，或者您可以使用 'cuda' 如果有 GPU
            self.q_net.eval()  # 设置为评估模式
            print(f"成功加载 DQN 模型: {self.model_path}")
        except FileNotFoundError:
            print(f"警告：DQN 模型文件未找到: {self.model_path}，请确保模型文件存在，或提供正确的路径。Agent 将使用随机策略或未训练模型。")
        except Exception as e:
            print(f"加载 DQN 模型时发生错误: {e}")

    def select_action(self, env):
        """
        使用 DQN 网络选择动作。

        Args:
            env (GomokuEnv): GomokuEnv 实例，提供棋盘状态和游戏环境信息。

        Returns:
            tuple: 选择的动作坐标 (x, y)，或 None 如果没有合法动作。
        """
        actions = self.selection_action_batch([env])
        return actions[0] if actions else None

    def get_evaluation_map(self, env):
        """
        返回基于 DQN 评估的评估 map。

        Args:
            env (GomokuEnv): GomokuEnv 实例，提供棋盘状态和游戏环境信息。

        Returns:
            dict: 评估 map，形如 {(i,j): {'offense': 0, 'defense': 0, 'combined': q_value}, ...}
        """
        evaluation_map = {}
        state_tensor = torch.tensor(env.board).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze()  # [board_size, board_size]
            
            # Create legal actions mask
            legal_actions = env.get_legal_actions()
            legal_mask = torch.zeros_like(q_values, dtype=torch.bool)
            legal_mask[torch.tensor([x[0] for x in legal_actions], dtype=torch.long),
                    torch.tensor([x[1] for x in legal_actions], dtype=torch.long)] = True
            
            # Mask illegal actions
            masked_q_values = torch.where(legal_mask, q_values, torch.tensor(0.).to(self.device))
            
            # Convert to evaluation map
            for x in range(env.board_size):
                for y in range(env.board_size):
                    if legal_mask[x, y]:
                        evaluation_map[(x, y)] = {
                            'combined': (masked_q_values[x, y].item() + 1) * 500,
                            'offense': 0,
                            'defense': 0
                        }
                    else:
                        evaluation_map[(x, y)] = {
                            'combined': 0,
                            'offense': 0,
                            'defense': 0
                        }
        
        return evaluation_map

    def selection_action_batch(self, envs):
        """
        批量选择下一步行动。

        Args:
            envs (list[GomokuEnv]): GomokuEnv 实例列表。

        Returns:
            list[tuple]: 选择的落子坐标列表 [(row, col), ...]。
        """
        if not envs:
            return []

        # Prepare batch input
        batch_states = torch.stack([
            torch.tensor(env.board).unsqueeze(0).float() 
            for env in envs
        ]).to(self.device)  # [batch_size, 1, board_size, board_size]

        with torch.no_grad():
            # Get Q-values for all states
            q_values = self.q_net(batch_states).squeeze(1)  # [batch_size, board_size, board_size]
            
            # Create legal actions mask for each environment
            legal_masks = torch.zeros_like(q_values, dtype=torch.bool)
            for i, env in enumerate(envs):
                legal_actions = env.get_legal_actions()
                if legal_actions:
                    legal_masks[i, torch.tensor([x[0] for x in legal_actions], dtype=torch.long),
                            torch.tensor([x[1] for x in legal_actions], dtype=torch.long)] = True
            
            # Mask illegal actions
            masked_q_values = torch.where(legal_masks, q_values, 
                                        torch.tensor(-1e6).to(self.device))
            
            # Get best legal actions
            best_idxs = torch.argmax(masked_q_values.view(len(envs), -1), dim=1)
            best_x = best_idxs // envs[0].board_size
            best_y = best_idxs % envs[0].board_size

        # Convert to list of tuples
        actions = [(int(x), int(y)) if m.any() else None 
                for x, y, m in zip(best_x, best_y, legal_masks)]
        
        return actions