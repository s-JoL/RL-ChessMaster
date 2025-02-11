import numpy as np
import torch
from agents.base_agent import BaseAgent  # 确保导入了 BaseAgent，即使 DQN Agent 可能不直接继承它
from networks.dqn_model import DQNNet  # 假设 DQNNet 定义在 networks.dqn_model.py 中

class DQNAgent(BaseAgent):
    """
    基于 DQN 的智能体。

    该 agent 使用训练好的 DQN 模型评估棋盘状态，
    并选择 Q 值最大的合法动作。
    与 RuleBasedAgent 保持接口一致，但内部使用 DQN 网络进行决策。
    """
    def __init__(self, model_path='./dqn_model.pth', q_net=None):
        """
        初始化 DQN Agent。

        Args:
            model_path (str): 训练好的 DQN 模型文件路径 (当 q_net=None 时使用).
            q_net (DQNNet, optional):  **直接传入的 DQN 模型实例。 如果提供，则忽略 model_path 加载。** 默认为 None.
        """
        self.model_path = model_path
        self._q_net_instance = q_net #  保存传入的 q_net 实例

        if q_net is not None: # **如果 q_net 参数被传入，则直接使用传入的模型**
            print("DQNAgent 初始化: 使用直接传入的 DQN 模型实例.")
            self.q_net = q_net #  使用传入的模型
            self.q_net.eval() # 设置为评估模式
        else: # **否则，仍然从 model_path 加载模型**
            self.q_net = DQNNet(board_size=board_size) #  初始化 DQN 网络 (如果 q_net 没有传入)
            self._load_model() # 加载预训练模型

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

    def select_action(self, env): # 修改输入为 env
        """
        使用 DQN 网络选择动作。

        Args:
            env (GomokuEnv): GomokuEnv 实例，提供棋盘状态和游戏环境信息。

        Returns:
            tuple: 选择的动作坐标 (x, y)，或 None 如果没有合法动作。
        """
        board = env.board # 从 env 中获取棋盘
        legal_actions = env.get_legal_actions() # 从 env 中获取合法动作
        if not legal_actions:
            return None # 没有合法动作

        state_tensor = torch.tensor(board).unsqueeze(0).unsqueeze(0).float() # 转换为 DQN 网络需要的输入格式 [1, 1, board_size, board_size]
        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze() #  [board_size, board_size]

        best_action = None
        best_q_value = -float('inf')

        for action in legal_actions:
            q_value = q_values[action[0], action[1]].item() # 获取该动作的 Q 值
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action # 返回 Q 值最大的合法动作

    def get_evaluation_map(self, env): # 修改输入为 env
        """
        返回基于 DQN 评估的评估 map。

        对于 DQN Agent，评估 map 可以理解为每个位置的 Q 值。
        返回一个形如： {(i,j): {'q_value': ...}, ...} 的字典。

        Args:
            env (GomokuEnv): GomokuEnv 实例，提供棋盘状态和游戏环境信息。

        Returns:
            dict: 评估 map，键为坐标 (x, y)，值为包含 'q_value' 键的字典。
        """
        evaluation_map = {}
        board = env.board # 从 env 中获取棋盘
        state_tensor = torch.tensor(board).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze() # [board_size, board_size]
        for x in range(env.board_size):
            for y in range(env.board_size):
                evaluation_map[(x, y)] = {'combined': q_values[x, y].item(), 'offense': 0, 'defense': 0} # 存储每个位置的 Q 值
        return evaluation_map
