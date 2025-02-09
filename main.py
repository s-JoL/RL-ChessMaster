"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 20:50:39
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-09 21:10:47
FilePath: /RL-ChessMaster/main.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import tkinter as tk
from tkinter import messagebox
from envs.gomoku_env import GomokuEnv
from agents.greedy_agent import GreedyAgent

BOARD_SIZE = 15  # 棋盘行列数

def score_to_color(score, min_score, max_score, base_color):
    """
    将分数映射为颜色：
      - base_color：当分数达到最大值时显示的颜色 (R, G, B)
      - 分数越低，颜色越偏向白色 (255,255,255)
    """
    if max_score == min_score:
        ratio = 1
    else:
        ratio = (score - min_score) / (max_score - min_score)
    r = int(255 * (1 - ratio) + base_color[0] * ratio)
    g = int(255 * (1 - ratio) + base_color[1] * ratio)
    b = int(255 * (1 - ratio) + base_color[2] * ratio)
    return f'#{r:02x}{g:02x}{b:02x}'

class GomokuGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("五子棋对局")
        
        # 状态变量：模式、编辑棋子、先手选择
        self.mode_var = tk.StringVar(value="standard")
        self.editing_mode = False
        self.edit_player_var = tk.IntVar(value=1)
        self.starting_player_var = tk.IntVar(value=1)
        
        # 创建顶部控制面板
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.create_control_panel()
        
        # ────────── 使用 grid 布局创建四个 Canvas ──────────
        # 顺序调整为：左上：棋盘；右上：综合评分；左下：进攻评分；右下：防守评分
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.board_canvas = tk.Canvas(self.canvas_frame, bg="beige")
        self.board_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.board_canvas.bind("<Button-1>", self.canvas_left_click)
        self.board_canvas.bind("<Button-3>", self.canvas_right_click)
        
        self.combined_canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.combined_canvas.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        # 初始标题，后续随尺寸调整更新
        self.combined_canvas.create_text(100, 10, text="综合分数", fill="black", font=("Arial", 12, "bold"), tags="title")
        
        self.offense_canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.offense_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.offense_canvas.create_text(100, 10, text="进攻分数", fill="black", font=("Arial", 12, "bold"), tags="title")
        
        self.defense_canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.defense_canvas.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.defense_canvas.create_text(100, 10, text="防守分数", fill="black", font=("Arial", 12, "bold"), tags="title")
        
        # 设置 grid 权重，使四个区域大小均分
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(1, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(1, weight=1)
        
        # 绑定 canvas_frame 的尺寸变化事件
        self.canvas_frame.bind("<Configure>", self.on_canvas_frame_resize)
        
        # 初始化环境与智能体
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.agent = GreedyAgent(board_size=BOARD_SIZE)
        self.human_player = 1   # 根据先手选择调整
        self.agent_player = -1  # 根据先手选择调整
        self.last_player = None
        
        self.draw_board_grid()
        self.initialize_board()
    
    def create_control_panel(self):
        """创建上部控制面板，包括模式选择、先手选择、编辑控件等"""
        tk.Label(self.control_frame, text="模式:").pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(self.control_frame, text="标准游戏", variable=self.mode_var, value="standard", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(self.control_frame, text="残局测试", variable=self.mode_var, value="endgame", command=self.on_mode_change).pack(side=tk.LEFT, padx=2)
        
        tk.Label(self.control_frame, text="先手玩家:").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(self.control_frame, text="人类（黑棋）", variable=self.starting_player_var, value=1).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(self.control_frame, text="AI（黑棋）", variable=self.starting_player_var, value=-1).pack(side=tk.LEFT, padx=2)
        
        # 编辑模式控件（仅在残局测试下显示）
        self.edit_frame = tk.Frame(self.control_frame)
        tk.Label(self.edit_frame, text="当前落子:").pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(self.edit_frame, text="黑子", variable=self.edit_player_var, value=1).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(self.edit_frame, text="白子", variable=self.edit_player_var, value=-1).pack(side=tk.LEFT, padx=2)
        self.clear_button = tk.Button(self.edit_frame, text="清空棋盘", command=self.clear_board)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.start_game_button = tk.Button(self.edit_frame, text="开始游戏", command=self.start_game_from_editing)
        self.start_game_button.pack(side=tk.LEFT, padx=5)
        self.edit_frame.pack_forget()
        
        self.restart_button = tk.Button(self.control_frame, text="重新开始", command=self.reset_game)
        self.restart_button.pack(side=tk.RIGHT, padx=5)
    
    def on_mode_change(self):
        if self.mode_var.get() == "endgame":
            self.editing_mode = True
            self.edit_frame.pack(side=tk.LEFT, padx=10)
            self.env.board[:] = 0
            self.update_board()
        else:
            self.editing_mode = False
            self.edit_frame.pack_forget()
            self.reset_game()
    
    def initialize_board(self):
        if self.mode_var.get() == "endgame":
            self.editing_mode = True
        else:
            self.reset_game()
    
    def reset_game(self):
        self.editing_mode = False
        self.env.reset()
        self.env.current_player = 1  # 固定黑棋先走
        if self.starting_player_var.get() == 1:
            self.human_player = 1
            self.agent_player = -1
        else:
            self.human_player = -1
            self.agent_player = 1
        self.last_player = None
        self.update_board()
        if self.env.get_current_player() == self.agent_player:
            self.master.after(500, self.agent_move)
    
    def start_game_from_editing(self):
        self.editing_mode = False
        self.env.current_player = 1
        if self.starting_player_var.get() == 1:
            self.human_player = 1
            self.agent_player = -1
        else:
            self.human_player = -1
            self.agent_player = 1
        messagebox.showinfo("提示", "残局设置完毕，游戏开始！")
        self.update_board()
        if self.env.get_current_player() == self.agent_player:
            self.master.after(500, self.agent_move)
    
    def clear_board(self):
        self.env.board[:] = 0
        self.update_board()
    
    def get_cell_size(self):
        """动态计算当前每个单元格的边长（基于棋盘 Canvas 的宽度）"""
        w = self.board_canvas.winfo_width()
        return w / BOARD_SIZE if w > 0 else 40
    
    def draw_board_grid(self):
        self.board_canvas.delete("grid")
        cell_size = self.get_cell_size()
        canvas_size = cell_size * BOARD_SIZE
        for i in range(BOARD_SIZE + 1):
            self.board_canvas.create_line(0, i * cell_size, canvas_size, i * cell_size, fill="gray", tags="grid")
        for j in range(BOARD_SIZE + 1):
            self.board_canvas.create_line(j * cell_size, 0, j * cell_size, canvas_size, fill="gray", tags="grid")
    
    def update_board(self):
        self.board_canvas.delete("stone")
        cell_size = self.get_cell_size()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.env.board[i, j] != 0:
                    self.draw_stone(i, j, self.env.board[i, j])
        self.draw_board_grid()
        self.update_evaluation_maps()
    
    def draw_stone(self, i, j, player):
        cell_size = self.get_cell_size()
        padding = cell_size * 0.125
        x0 = j * cell_size + padding
        y0 = i * cell_size + padding
        x1 = (j + 1) * cell_size - padding
        y1 = (i + 1) * cell_size - padding
        color = "black" if player == 1 else "white"
        self.board_canvas.create_oval(x0, y0, x1, y1, fill=color, tags="stone")
    
    def draw_evaluation_grid(self, canvas):
        canvas.delete("grid")
        cell_size = canvas.winfo_width() / BOARD_SIZE if canvas.winfo_width() > 0 else 40
        canvas_size = cell_size * BOARD_SIZE
        for i in range(BOARD_SIZE + 1):
            canvas.create_line(0, i * cell_size, canvas_size, i * cell_size, fill="gray", tags="grid")
        for j in range(BOARD_SIZE + 1):
            canvas.create_line(j * cell_size, 0, j * cell_size, canvas_size, fill="gray", tags="grid")
    
    def update_evaluation_maps(self):
        # 删除旧绘制项（不包括标题）
        self.offense_canvas.delete("cell")
        self.defense_canvas.delete("cell")
        self.combined_canvas.delete("cell")
        self.offense_canvas.delete("grid")
        self.defense_canvas.delete("grid")
        self.combined_canvas.delete("grid")
        
        cell_size = self.get_cell_size()
        canvas_size = cell_size * BOARD_SIZE
        
        # 获取评估数据（此处以 human_player 为依据，可根据需要调整）
        score_map = self.agent.get_evaluation_map(self.env.board, self.human_player)
        
        offense_vals = [v['offense'] for v in score_map.values()]
        defense_vals = [v['defense'] for v in score_map.values()]
        combined_vals = [v['combined'] for v in score_map.values()]
        off_min, off_max = (min(offense_vals), max(offense_vals)) if offense_vals else (0, 1)
        def_min, def_max = (min(defense_vals), max(defense_vals)) if defense_vals else (0, 1)
        comb_min, comb_max = (min(combined_vals), max(combined_vals)) if combined_vals else (0, 1)
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                x0 = j * cell_size
                y0 = i * cell_size
                x1 = (j + 1) * cell_size
                y1 = (i + 1) * cell_size
                if self.env.board[i, j] != 0:
                    cell_color = "lightgray"
                    self.offense_canvas.create_rectangle(x0, y0, x1, y1, fill=cell_color, outline="", tags="cell")
                    self.defense_canvas.create_rectangle(x0, y0, x1, y1, fill=cell_color, outline="", tags="cell")
                    self.combined_canvas.create_rectangle(x0, y0, x1, y1, fill=cell_color, outline="", tags="cell")
                else:
                    if (i, j) in score_map:
                        off_score = score_map[(i, j)]['offense']
                        def_score = score_map[(i, j)]['defense']
                        comb_score = score_map[(i, j)]['combined']
                    else:
                        off_score = def_score = comb_score = 0
                    off_color = score_to_color(off_score, off_min, off_max, base_color=(255, 0, 0))
                    def_color = score_to_color(def_score, def_min, def_max, base_color=(0, 0, 255))
                    comb_color = score_to_color(comb_score, comb_min, comb_max, base_color=(0, 255, 0))
                    
                    self.offense_canvas.create_rectangle(x0, y0, x1, y1, fill=off_color, outline="", tags="cell")
                    self.defense_canvas.create_rectangle(x0, y0, x1, y1, fill=def_color, outline="", tags="cell")
                    self.combined_canvas.create_rectangle(x0, y0, x1, y1, fill=comb_color, outline="", tags="cell")
                    
                    # 在每个单元格中心显示数值，字体大小随 cell_size 调整
                    font_size = max(int(cell_size / 5), 8)
                    self.offense_canvas.create_text((x0+x1)//2, (y0+y1)//2, text=f"{off_score:.0f}",
                                                     font=("Arial", font_size), tags="cell")
                    self.defense_canvas.create_text((x0+x1)//2, (y0+y1)//2, text=f"{def_score:.0f}",
                                                     font=("Arial", font_size), tags="cell")
                    self.combined_canvas.create_text((x0+x1)//2, (y0+y1)//2, text=f"{comb_score:.0f}",
                                                     font=("Arial", font_size), tags="cell")
        self.draw_evaluation_grid(self.offense_canvas)
        self.draw_evaluation_grid(self.defense_canvas)
        self.draw_evaluation_grid(self.combined_canvas)
    
    def on_canvas_frame_resize(self, event):
        # 根据 canvas_frame 的新尺寸计算每个 Canvas 应该使用的边长：
        # 每个 Canvas 占据 canvas_frame 的一半宽度和一半高度，取较小者以保证正方形
        new_cell_size = min(event.width // 2, event.height // 2) / BOARD_SIZE
        new_size = new_cell_size * BOARD_SIZE
        # 调整所有 Canvas 的大小
        self.board_canvas.config(width=new_size, height=new_size)
        self.combined_canvas.config(width=new_size, height=new_size)
        self.offense_canvas.config(width=new_size, height=new_size)
        self.defense_canvas.config(width=new_size, height=new_size)
        self.update_titles(new_size)
        self.update_board()
    
    def update_titles(self, new_size):
        self.combined_canvas.delete("title")
        self.combined_canvas.create_text(new_size/2, 10, text="综合分数", fill="black", font=("Arial", 12, "bold"), tags="title")
        self.offense_canvas.delete("title")
        self.offense_canvas.create_text(new_size/2, 10, text="进攻分数", fill="black", font=("Arial", 12, "bold"), tags="title")
        self.defense_canvas.delete("title")
        self.defense_canvas.create_text(new_size/2, 10, text="防守分数", fill="black", font=("Arial", 12, "bold"), tags="title")
    
    def canvas_left_click(self, event):
        if self.editing_mode:
            self.on_edit_left_click(event)
        else:
            self.on_game_click(event)
    
    def canvas_right_click(self, event):
        if self.editing_mode:
            self.on_edit_right_click(event)
    
    def on_edit_left_click(self, event):
        cell_size = self.get_cell_size()
        j = int(event.x // cell_size)
        i = int(event.y // cell_size)
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            self.env.board[i, j] = self.edit_player_var.get()
            self.update_board()
    
    def on_edit_right_click(self, event):
        cell_size = self.get_cell_size()
        j = int(event.x // cell_size)
        i = int(event.y // cell_size)
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            self.env.board[i, j] = 0
            self.update_board()
    
    def on_game_click(self, event):
        if self.env.done:
            return
        if self.env.get_current_player() != self.human_player:
            return
        cell_size = self.get_cell_size()
        j = int(event.x // cell_size)
        i = int(event.y // cell_size)
        if not (0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE):
            return
        if self.env.board[i, j] != 0:
            return
        self.last_player = self.env.get_current_player()
        action = i * BOARD_SIZE + j
        board, reward, done = self.env.step(action)
        self.update_board()
        if done:
            self.show_result(reward)
        else:
            self.master.after(500, self.agent_move)
    
    def agent_move(self):
        if self.env.done:
            return
        if self.env.get_current_player() == self.agent_player:
            move = self.agent.select_action(self.env.board, self.agent_player)
            if move is None:
                self.show_message("平局！")
                return
            self.last_player = self.env.get_current_player()
            action = move[0] * BOARD_SIZE + move[1]
            board, reward, done = self.env.step(action)
            self.update_board()
            if done:
                self.show_result(reward)
    
    def show_result(self, reward):
        if reward == 1:
            msg = "恭喜，你赢了！" if self.last_player == self.human_player else "很遗憾，AI 获胜！"
        elif reward == -1:
            msg = "非法落子，游戏结束！"
        else:
            msg = "平局！"
        messagebox.showinfo("游戏结束", msg)
    
    def show_message(self, msg):
        messagebox.showinfo("提示", msg)

def main():
    root = tk.Tk()
    gui = GomokuGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
