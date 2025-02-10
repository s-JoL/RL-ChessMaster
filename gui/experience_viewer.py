"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-11 01:40:34
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-11 01:46:34
FilePath: /RL-ChessMaster/gui/experience_viewer.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import tkinter as tk
from tkinter import ttk, messagebox

class ExperienceViewer(tk.Tk):
    def __init__(self, experience_batch):
        super().__init__()
        self.experience_batch = experience_batch
        self.current_index = 0
        self._setup_ui()
        self.update_display()

    def _setup_ui(self):
        self.title("Experience Pool Viewer")
        self.board_size = 15
        self.cell_size = 30
        self.piece_radius = 12

        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.canvas = tk.Canvas(main_frame, 
                              width=self.board_size * self.cell_size,
                              height=self.board_size * self.cell_size,
                              bg='gray92')
        self.canvas.grid(row=1, column=0, columnspan=2)

        self._create_info_labels(main_frame)
        self._create_navigation_buttons()

    def _create_info_labels(self, parent):
        labels = [
            ("Your Action (Red):", "your_action_label"),
            ("Opponent Action (Blue):", "opponent_action_label"),
            ("Reward:", "reward_label"),
            ("Game Step:", "game_step_label")
        ]
        
        # 使用单独的Frame来组织标签
        info_frame = ttk.Frame(parent)
        info_frame.grid(row=0, column=0, sticky=tk.W, pady=10)
        
        for i, (text, attr) in enumerate(labels):
            # 使用grid的padx参数增加水平间距
            ttk.Label(info_frame, text=text).grid(row=i, column=0, sticky=tk.W, padx=5)
            setattr(self, attr, ttk.Label(info_frame, text=""))
            getattr(self, attr).grid(row=i, column=1, sticky=tk.W, padx=5)

    def _create_navigation_buttons(self):
        button_frame = ttk.Frame(self, padding="10")
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

        ttk.Button(button_frame, text="Previous", command=self.prev_experience).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(button_frame, text="Next", command=self.next_experience).grid(row=0, column=1, sticky=tk.E)

        self.current_index_label = ttk.Label(button_frame, text=f"Experience {self.current_index + 1}/{len(self.experience_batch)}")
        self.current_index_label.grid(row=0, column=2, padx=10)

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
