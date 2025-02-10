import random
import multiprocessing
from collections import deque
from tqdm import tqdm  # 进度条库
from envs.gomoku_env import GomokuEnv
from agents.greedy_agent import GreedyAgent

class BalancedMemoryBank:
    def __init__(self, max_size=10000, num_bins=15, max_steps=225):
        """
        :param max_size: 总的 memory bank 容量
        :param num_bins: 将步数（残局阶段）分为多少个 bin
        :param max_steps: 一局游戏的最大步数（用于划分 bins 的参考值）
        """
        self.max_size = max_size
        self.num_bins = num_bins
        self.bin_capacity = max_size // num_bins  # 每个 bin 的最大容量
        self.max_steps = max_steps
        # 用字典维护每个 bin 的数据，key 为 bin 的索引，value 为 deque 结构
        self.bins = {i: deque(maxlen=self.bin_capacity) for i in range(num_bins)}

    def get_bin_index(self, step):
        """
        根据当前步数映射到对应的 bin 索引。
        假设最大步数为 max_steps，将 [0, max_steps] 均分为 num_bins 个区间。
        """
        bin_size = self.max_steps / self.num_bins
        bin_idx = int(step // bin_size)
        if bin_idx >= self.num_bins:
            bin_idx = self.num_bins - 1
        return bin_idx

    def add(self, experience):
        """
        添加经验。experience 是一个元组：(state, action, reward, next_state, step)
        根据 step 将经验存入对应的 bin，如果该 bin 已满，则自动删除最旧的记录。
        """
        step = experience[-1]  # 最后一个元素为步数
        bin_idx = self.get_bin_index(step)
        self.bins[bin_idx].append(experience)

    def sample(self, batch_size):
        """
        从各个 bin 中均匀采样，保证每个阶段的数据都有涉及。
        如果某个 bin 内样本不足，则采样该 bin 已有的所有样本，并从所有经验中补足。
        """
        samples = []
        per_bin = max(1, batch_size // self.num_bins)
        for i in range(self.num_bins):
            bin_data = list(self.bins[i])
            if bin_data:
                if len(bin_data) >= per_bin:
                    samples.extend(random.sample(bin_data, per_bin))
                else:
                    samples.extend(bin_data)
        if len(samples) < batch_size:
            all_experiences = []
            for bin_data in self.bins.values():
                all_experiences.extend(list(bin_data))
            if len(all_experiences) >= batch_size:
                samples = random.sample(all_experiences, batch_size)
            else:
                samples = all_experiences
        return samples

    def size(self):
        """返回所有 bin 内存储的经验总数"""
        return sum(len(bin_data) for bin_data in self.bins.values())

def simulate_single_game(args):
    """
    模拟一局游戏，返回该局游戏所有步的经验。
    与之前不同的是：
      - 每一轮先模拟自己的行动，再模拟对手的行动，
      - 记录的 next_state 包含了对手的落子，
      - agent 的净奖励为：reward_agent - reward_opponent，
        即如果对手的行动获得奖励（例如胜利），则自己应受到惩罚。
    
    :param args: 二元组 (p_random, board_size)
    :return: 该局游戏的经验列表，每个经验为 (state, action, net_reward, next_state, step)
    """
    p_random, board_size = args
    local_experiences = []
    env = GomokuEnv(board_size=board_size)
    # 这里对双方均使用 GreedyAgent（当然你也可以换成不同的对手策略）
    agent = GreedyAgent(board_size=board_size)
    
    state = env.reset()  # 重置游戏
    done = False
    step_count = 0

    while not done:
        legal_actions = env.get_legal_actions()  # 获取当前所有合法动作
        if len(legal_actions) < 10:
            break
        # 自己（当前玩家）的行动
        if random.random() < p_random:
            action = random.choice(legal_actions)
        else:
            action = agent.select_action(state, env.get_current_player())
        # 模拟自己的落子
        intermediate_state, reward_agent, done = env.step(action)
        # 如果自己的行动直接终结了游戏，则记录经验后退出
        if done:
            local_experiences.append((state, action, reward_agent, intermediate_state, step_count))
            break
        
        # 对手的回合：对手也按贪心策略选择动作
        opponent_legal_actions = env.get_legal_actions()
        if len(opponent_legal_actions) < 1:
            break
        opponent_action = agent.select_action(intermediate_state, env.get_current_player())
        next_state, reward_opponent, done = env.step(opponent_action)
        # 计算净奖励：自己的奖励减去对手获得的奖励（对手奖励对自己来说是负面效果）
        net_reward = reward_agent - reward_opponent
        # 记录经验，next_state 已包含对手的落子
        local_experiences.append((state, action, net_reward, next_state, step_count))
        state = next_state
        step_count += 1

    return local_experiences

def generate_initial_memory_multiprocess(memory_bank, total_games=3000, p_random=0.1, num_processes=4, board_size=15):
    """
    使用多进程生成初始 memory bank 数据：
      - total_games: 总共模拟的游戏局数
      - num_processes: 使用的进程数量
    每个任务模拟一局游戏，返回该局游戏的所有经验，主进程收集后将其添加到 memory bank 中。
    同时使用 tqdm 显示进度条。
    """
    # 构造任务列表：每个任务的参数为 (p_random, board_size)
    tasks = [(p_random, board_size)] * total_games
    all_experiences = []

    # 使用 multiprocessing.Pool 分发任务
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 imap 可以边获取结果边更新进度条
        for experiences in tqdm(pool.imap(simulate_single_game, tasks),
                                total=total_games,
                                desc="Generating memory bank"):
            all_experiences.extend(experiences)

    # 将所有采集到的经验添加到 memory bank 中
    for exp in all_experiences:
        memory_bank.add(exp)

if __name__ == '__main__':
    # 初始化全局 memory bank
    balanced_memory_bank = BalancedMemoryBank(max_size=10000, num_bins=5, max_steps=225)

    # 使用多进程生成初始 memory bank 数据，同时显示进度条
    generate_initial_memory_multiprocess(
        memory_bank=balanced_memory_bank,
        total_games=3000,
        p_random=0.1,
        num_processes=4,   # 可根据 CPU 核心数进行调整
        board_size=15
    )

    print("Balanced Memory Bank Size:", balanced_memory_bank.size())
