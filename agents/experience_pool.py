"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-10 23:47:06
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-11 01:30:52
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
        """
        从经验池中随机采样一个批次的经验，确保不同步数的经验均衡。

        Args:
            batch_size (int): 采样的批次大小。

        Returns:
            list: 包含字典的列表，每个字典代表一条经验。
        """
        # 获取所有非空的桶
        non_empty_buckets = [step for step, experiences in self.experience_buckets.items() 
                           if experiences]
        
        if not non_empty_buckets:
            return []

        # 计算每个桶应该采样的数量
        samples_per_bucket = max(1, batch_size // len(non_empty_buckets))
        batch = []
        
        # 随机化桶的顺序
        random.shuffle(non_empty_buckets)

        # 从每个桶中采样
        for step in non_empty_buckets:
            bucket = self.experience_buckets[step]
            num_samples = min(samples_per_bucket, len(bucket))
            batch.extend(random.sample(bucket, num_samples))

        # 如果样本不足，随机补充
        while len(batch) < batch_size and non_empty_buckets:
            step = random.choice(non_empty_buckets)
            bucket = self.experience_buckets[step]
            if bucket:
                batch.append(random.choice(bucket))

        random.shuffle(batch)
        return batch[:batch_size]

    def _generate_initial_experiences(self, num_experiences):
        """
        生成指定数量的初始经验数据。在一个进程中运行。
        """
        experiences = []
        # 假设平均每局游戏产生1步
        avg_steps_per_game = 1
        games_needed = (num_experiences + avg_steps_per_game - 1) // avg_steps_per_game  # 向上取整
        
        env = GomokuEnv(board_size=self.board_size)
        agent = GreedyAgent(
            board_size=self.board_size,
            win_length=self.win_length,
            offense_weight=self.offense_weight,
            defense_weight=self.defense_weight
        )

        for game_id in range(games_needed):
            env.reset()
            game_step = 0
            done = False
            game_states = []  # 存储一局游戏的所有状态

            while not done and game_step < self.max_game_steps:
                # Agent's turn
                state = env.board.copy()
                
                # 有20%的概率采取随机策略
                if random.random() < 0.2:
                    empty_positions = list(zip(*np.where(state == 0)))
                    if empty_positions:
                        action = random.choice(empty_positions)
                    else:
                        action = None
                else:
                    action = agent.select_action(state, env.get_current_player())
                
                if action is None:
                    break

                board_after_move, reward_agent, done_agent = env.step(action)
                
                # 对手的回合
                opponent_action = None
                if not done_agent:
                    # 对手也有20%的概率采取随机策略
                    if random.random() < 0.2:
                        empty_positions = list(zip(*np.where(board_after_move == 0)))
                        if empty_positions:
                            opponent_action = random.choice(empty_positions)
                        else:
                            opponent_action = None
                    else:
                        opponent_action = agent.select_action(board_after_move, env.get_current_player())
                    
                    if opponent_action is not None:
                        next_state, reward_opponent, done_opponent = env.step(opponent_action)
                    else:
                        next_state, done_opponent = board_after_move, True
                else:
                    next_state, done_opponent = board_after_move, True

                # 计算奖励
                reward = 100 if done_agent else (-100 if done_opponent else 0)
                
                game_states.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'game_step': game_step,
                    'opponent_action': opponent_action,
                    'game_id': game_id
                })

                game_step += 1
                done = done_agent or done_opponent

            experiences.extend(game_states)

        return experiences[:]  # 只返回需要的数量

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

        processes = []
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
    import tkinter as tk
    from tkinter import ttk, messagebox
    import numpy as np

    experience_pool = ExperiencePool(capacity=50)
    print("初始化前经验池大小:", sum(len(bucket) for bucket in experience_pool.experience_buckets.values()))
    experience_pool.initialize_pool(initial_size=100)
    print("初始化后经验池大小:", sum(len(bucket) for bucket in experience_pool.experience_buckets.values()))

    batch = experience_pool.sample_experience_batch(batch_size=100)
    if not batch:
        print("经验池为空，无法采样展示。请检查经验池初始化是否成功。")
    else:
        class ExperienceViewer(tk.Tk):
            def __init__(self, experience_batch):
                super().__init__()
                self.experience_batch = experience_batch
                self.current_index = 0
                self.title("Experience Pool Viewer")
                self.board_size = experience_pool.board_size
                self.cell_size = 30
                self.piece_radius = 12

                # 主显示框架
                main_frame = ttk.Frame(self, padding="10")
                main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

                # 棋盘画布
                self.canvas = tk.Canvas(main_frame, 
                                      width=self.board_size * self.cell_size,
                                      height=self.board_size * self.cell_size,
                                      bg='gray92')
                self.canvas.grid(row=1, column=0, columnspan=2)

                # 信息显示区域
                info_frame = ttk.Frame(self, padding="10")
                info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

                ttk.Label(info_frame, text="Your Action (Red):").grid(row=0, column=0, sticky=tk.W)
                self.your_action_label = ttk.Label(info_frame, text="")
                self.your_action_label.grid(row=0, column=1, sticky=tk.W)

                ttk.Label(info_frame, text="Opponent Action (Blue):").grid(row=1, column=0, sticky=tk.W)
                self.opponent_action_label = ttk.Label(info_frame, text="")
                self.opponent_action_label.grid(row=1, column=1, sticky=tk.W)

                ttk.Label(info_frame, text="Reward:").grid(row=2, column=0, sticky=tk.W)
                self.reward_label = ttk.Label(info_frame, text="")
                self.reward_label.grid(row=2, column=1, sticky=tk.W)

                ttk.Label(info_frame, text="Game Step:").grid(row=3, column=0, sticky=tk.W)
                self.game_step_label = ttk.Label(info_frame, text="")
                self.game_step_label.grid(row=3, column=1, sticky=tk.W)

                # 导航按钮
                button_frame = ttk.Frame(self, padding="10")
                button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

                ttk.Button(button_frame, text="Previous", command=self.prev_experience).grid(row=0, column=0, sticky=tk.W)
                ttk.Button(button_frame, text="Next", command=self.next_experience).grid(row=0, column=1, sticky=tk.E)

                self.current_index_label = ttk.Label(button_frame, text=f"Experience {self.current_index + 1}/{len(self.experience_batch)}")
                self.current_index_label.grid(row=0, column=2, padx=10)

                self.update_display()

            def update_display(self):
                experience = self.experience_batch[self.current_index]
                state = experience['next_state']  # 使用next_state来显示双方落子后的状态
                action = experience['action']
                opponent_action = experience['opponent_action']
                game_step = experience['game_step']
                game_id = experience.get('game_id', -1)

                self.title(f"Experience Pool Viewer - Game {game_id}")
                self.draw_board(state, action, opponent_action)

                self.your_action_label.config(text=str(action))
                self.opponent_action_label.config(text=str(opponent_action) if opponent_action else "None")
                self.reward_label.config(text=str(experience['reward']))
                self.game_step_label.config(text=str(game_step))
                self.current_index_label.config(text=f"Experience {self.current_index + 1}/{len(self.experience_batch)}")

            def draw_board(self, board, action, opponent_action):
                self.canvas.delete("all")
                # 绘制网格线
                for i in range(self.board_size):
                    self.canvas.create_line(i * self.cell_size, 0, 
                                          i * self.cell_size, self.board_size * self.cell_size, 
                                          fill="black")
                    self.canvas.create_line(0, i * self.cell_size, 
                                          self.board_size * self.cell_size, i * self.cell_size, 
                                          fill="black")

                # 绘制棋子
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        piece = board[row][col]
                        if piece == 1:
                            self.draw_piece(col, row, "black")
                        elif piece == -1:
                            self.draw_piece(col, row, "white")

                # 用红色标记自己的动作
                if action:
                    action_x, action_y = action
                    self.draw_highlight(action_y, action_x, 'red')

                # 用蓝色标记对手的动作
                if opponent_action:
                    opp_x, opp_y = opponent_action
                    self.draw_highlight(opp_y, opp_x, 'blue')

            def draw_piece(self, col, row, color):
                x_center = col * self.cell_size + self.cell_size // 2
                y_center = row * self.cell_size + self.cell_size // 2
                x0 = x_center - self.piece_radius
                y0 = y_center - self.piece_radius
                x1 = x_center + self.piece_radius
                y1 = y_center + self.piece_radius
                self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="black")

            def draw_highlight(self, col, row, color):
                x_center = col * self.cell_size + self.cell_size // 2
                y_center = row * self.cell_size + self.cell_size // 2
                x0 = x_center - self.piece_radius - 2
                y0 = y_center - self.piece_radius - 2
                x1 = x_center + self.piece_radius + 2
                y1 = y_center + self.piece_radius + 2
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)

            def next_experience(self):
                if self.current_index < len(self.experience_batch) - 1:
                    self.current_index += 1
                    self.update_display()
                else:
                    messagebox.showinfo("End", "已经到达最后一个经验。")

            def prev_experience(self):
                if self.current_index > 0:
                    self.current_index -= 1
                    self.update_display()
                else:
                    messagebox.showinfo("Start", "已经到达第一个经验。")

        app = ExperienceViewer(batch)
        app.mainloop()