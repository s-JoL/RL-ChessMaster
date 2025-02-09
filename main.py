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

# 每个格子的像素大小和棋盘尺寸
CELL_SIZE = 40
BOARD_SIZE = 15

class GomokuGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("五子棋对局")
        self.canvas = tk.Canvas(master, width=BOARD_SIZE * CELL_SIZE, height=BOARD_SIZE * CELL_SIZE, bg="beige")
        self.canvas.pack(side=tk.TOP)
        
        # 增加“重新开始”按钮
        self.reset_button = tk.Button(master, text="重新开始", command=self.reset_game)
        self.reset_button.pack(side=tk.BOTTOM, pady=10)
        
        # 绑定鼠标点击事件
        self.canvas.bind("<Button-1>", self.on_click)
        
        # 初始化环境和智能体
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.agent = GreedyAgent(board_size=BOARD_SIZE)
        
        # 选择执子：此处固定人类为 1（黑棋），AI 为 -1（白棋）
        # 如果你需要让玩家选择先后手，可在此处添加选择对话框
        self.human_player = 1
        self.agent_player = -1
        
        # 用于记录最后一次落子方（在获胜判断时使用）
        self.last_player = None

        # 绘制棋盘网格
        self.draw_board()
        self.reset_game()

    def reset_game(self):
        """重置游戏状态，清空棋盘并重新开始"""
        self.env.reset()
        self.last_player = None
        # 清除画布中已有的棋子和网格，再重新绘制
        self.canvas.delete("all")
        self.draw_board()
        self.update_board()
        # 如果AI先手，则延时调用 AI 落子
        if self.env.get_current_player() == self.agent_player:
            self.master.after(500, self.agent_move)

    def draw_board(self):
        """绘制棋盘网格"""
        for i in range(BOARD_SIZE):
            # 水平线
            self.canvas.create_line(0, i * CELL_SIZE, BOARD_SIZE * CELL_SIZE, i * CELL_SIZE, fill="gray")
            # 垂直线
            self.canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, BOARD_SIZE * CELL_SIZE, fill="gray")

    def update_board(self):
        """根据环境状态绘制棋子"""
        # 清除已有棋子（tag 为 "stone" 的图形）
        self.canvas.delete("stone")
        board = self.env.board
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] != 0:
                    self.draw_stone(i, j, board[i, j])

    def draw_stone(self, i, j, player):
        """
        在棋盘位置 (i,j) 绘制棋子
        :param i: 行索引
        :param j: 列索引
        :param player: 棋子所属玩家，1 显示为黑子，-1 显示为白子
        """
        color = "black" if player == 1 else "white"
        x0 = j * CELL_SIZE + 5
        y0 = i * CELL_SIZE + 5
        x1 = (j + 1) * CELL_SIZE - 5
        y1 = (i + 1) * CELL_SIZE - 5
        self.canvas.create_oval(x0, y0, x1, y1, fill=color, tags="stone")

    def on_click(self, event):
        """处理玩家点击事件：转换点击坐标到棋盘位置并尝试下子"""
        if self.env.done:
            return  # 游戏结束后不再处理点击

        # 若当前不是人类回合，则忽略点击
        if self.env.get_current_player() != self.human_player:
            return

        # 根据鼠标点击坐标计算棋盘格子（行 i，列 j）
        j = event.x // CELL_SIZE
        i = event.y // CELL_SIZE
        if i < 0 or i >= BOARD_SIZE or j < 0 or j >= BOARD_SIZE:
            return
        if self.env.board[i, j] != 0:
            return  # 此位置已有棋子，忽略

        self.last_player = self.env.get_current_player()
        action = i * BOARD_SIZE + j
        board, reward, done = self.env.step(action)
        self.update_board()

        if done:
            self.show_result(reward)
        else:
            # 延时让 AI 落子
            self.master.after(500, self.agent_move)

    def agent_move(self):
        """由 AI 落子"""
        if self.env.done:
            return

        # 如果当前正好轮到 AI
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
        """游戏结束后弹窗提示结果"""
        if reward == 1:
            # 根据最后落子的玩家判断胜负
            if self.last_player == self.human_player:
                msg = "恭喜，你赢了！"
            else:
                msg = "很遗憾，AI 获胜！"
        elif reward == -1:
            msg = "非法落子，游戏结束！"
        else:
            msg = "平局！"
        messagebox.showinfo("游戏结束", msg)

    def show_message(self, msg):
        """显示提示信息"""
        messagebox.showinfo("提示", msg)

def main():
    root = tk.Tk()
    gui = GomokuGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
