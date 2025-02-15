"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 20:50:26
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-15 22:41:28
FilePath: /RL-ChessMaster/utils/board_utils.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import numpy as np

def check_winner(board, x, y, win_length=5):
    """
    检查在棋盘 (x, y) 位置落子后，是否出现获胜方。

    参数:
      board: 当前棋盘 (numpy 数组).
      x, y:  最后落子的坐标.
      win_length: 获胜需要的连子数 (默认 5).

    返回值:
      bool: 如果有玩家获胜则返回 True, 否则返回 False.
    """
    rows, cols = board.shape
    if rows < win_length or cols < win_length:
        raise ValueError(f"Board size must be at least {win_length}x{win_length}")

    player = board[x, y]
    if player == 0:
        return False

    directions = [((1, 0), (-1, 0)), ((0, 1), (0, -1)), ((1, 1), (-1, -1)), ((1, -1), (-1, 1))]
    for d1, d2 in directions:
        count = 1
        for dx, dy in (d1, d2):
            curr_x, curr_y = x + dx, y + dy
            while 0 <= curr_x < rows and 0 <= curr_y < cols and board[curr_x, curr_y] == player:
                count += 1
                if count >= win_length:
                    return True
                curr_x += dx
                curr_y += dy
    return False

def is_alive_pattern(board, x, y, length):
    """
    检查在棋盘 (x, y) 位置是否形成指定长度的活棋型 (两端都为空位).

    参数:
      board: 当前棋盘 (numpy 数组).
      x, y:  待检查的位置坐标.
      length: 需要检查的棋子长度.

    返回值:
      bool: 如果是活棋型则返回 True, 否则返回 False.
    """
    rows, cols = board.shape
    player = board[x, y]
    directions = [((1, 0), (-1, 0)), ((0, 1), (0, -1)), ((1, 1), (-1, -1)), ((1, -1), (-1, 1))]
    for d1, d2 in directions:
        count = 1
        open_ends = 0
        for dx, dy in (d1, d2):
            curr_x, curr_y = x, y
            for _ in range(length - 1):
                curr_x += dx
                curr_y += dy
                if 0 <= curr_x < rows and 0 <= curr_y < cols and board[curr_x, curr_y] == player:
                    count += 1
                else:
                    break
            if 0 <= curr_x < rows and 0 <= curr_y < cols and board[curr_x, curr_y] == 0:
                open_ends += 1
        if count >= length and open_ends == 2:
            return True
    return False

def analyze_direction(board, x, y, color, directions):
    """
    分析指定位置在给定方向上的棋型，返回连续子数、跳子 bonus、封堵数等信息。

    参数:
      board: 当前棋盘 (numpy 数组).
      x, y:  待分析的位置坐标.
      color:  棋子颜色 (1 或 -1).
      directions:  要分析的方向列表，每个方向为一对元组，如 [(dx1, dy1), (dx2, dy2)].

    返回值:
      tuple: (pure_count, bonus, total_blocks, jump_used)
             - pure_count:  连续同色棋子的数量 (包含当前位置).
             - bonus:  跳子 bonus (0, 0.5 或 1).
             - total_blocks:  两侧被封堵的个数 (0 表示活型).
             - jump_used:  是否使用了跳子.
    """
    rows, cols = board.shape
    cont = [0, 0]
    jump = [False, False]
    block = [0, 0]

    for i, (dx, dy) in enumerate(directions):
        step = 1
        while True:
            nx = x + dx * step
            ny = y + dy * step
            if not (0 <= nx < rows and 0 <= ny < cols):
                block[i] = 1
                break
            cell = board[nx, ny]
            if cell == color:
                cont[i] += 1
                step += 1
            else:
                if cell != 0:
                    block[i] = 1
                break
        
        if cont[i] == 0:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and board[nx, ny] == 0:
                nx2, ny2 = x + dx * 2, y + dy * 2
                if 0 <= nx2 < rows and 0 <= ny2 < cols and board[nx2, ny2] == color:
                    jump[i] = True

    pure_count = 1 + cont[0] + cont[1]
    total_blocks = block[0] + block[1]
    jump_used = jump[0] or jump[1]
    bonus = 1 if jump[0] and jump[1] else (0.5 if (jump[0] or jump[1]) and (cont[0] + cont[1] > 0) else (1 if jump[0] or jump[1] else 0))
    return pure_count, bonus, total_blocks, jump_used

def calculate_position_weight(board_size, x, y):
    """
    计算位置权重，靠近中心的位置权重更高。
    """
    center = board_size // 2
    distance = abs(x - center) + abs(y - center)
    max_distance = 2 * center
    weight = 1 - (distance / max_distance) * 0.5
    return weight

# ────────── 以下为评估空位的函数 ──────────

# 基础分值表与跳子 bonus 因子
score_table = {
    'win5': 100000,    # 连五
    'alive4': 10000,   # 活四
    'double3': 5000,   # 双活三
    'dead4': 1000,     # 死四
    'alive3': 500,     # 活三
    'dead3': 200,      # 死三
    'alive2': 100,     # 活二
    'dead2': 50        # 死二
}
bonus_factor = 100

def compute_score(board, x, y, target, win_length=5, mode="offense"):
    """
    计算在 (x,y) 落子后，对 target 方（可为我方或对手）的评分，
    mode 为 "offense" 表示进攻评分（target 为我方），
    mode 为 "defense" 表示防守评分（target 为对手）。
    
    评分依据各个方向上连续棋子的数量、棋型（活型或死型）以及跳子 bonus 计算，
    其中在进攻模式下还会统计“双活三”数量以给予额外加分。
    
    参数:
      board: 当前棋盘（numpy 数组）
      x, y: 待评估位置
      target: 需要计算的棋子类型（进攻时为我方棋子，防守时为对手棋子）
      win_length: 获胜所需连子数（默认 5）
      mode: "offense" 或 "defense"
      
    返回:
      分数（数值）
    """
    directions = [
        [(0, 1), (0, -1)],    # 水平
        [(1, 0), (-1, 0)],    # 垂直
        [(1, 1), (-1, -1)],   # 主对角线
        [(1, -1), (-1, 1)]    # 副对角线
    ]
    
    total_score = 0
    alive_three_count = 0
    # 当 mode 为防守时，使用防守因子；进攻时因子为 1.0
    factor = 0.8 if mode == "defense" else 1.0
    
    for direction_pair in directions:
        # analyze_direction 在每个方向返回：
        #   pure: 连续同色棋子的数量（包含当前位置）
        #   bonus: 跳子 bonus（可能为 0、0.5 或 1）
        #   blocks: 两侧被封堵的个数（0 表示活型）
        #   jump_used: 是否使用了跳子
        pure, bonus, blocks, jump_used = analyze_direction(board, x, y, target, direction_pair)
        direction_score = 0.1
        
        if blocks == 0:  # 活棋型
            if pure >= win_length:
                direction_score += score_table['win5']
            elif pure == win_length - 1:
                direction_score += score_table['alive4']
            elif pure == win_length - 2:
                direction_score += score_table['alive3']
                # 在进攻时统计“双活三”（且要求没有用跳子辅助）
                if mode == "offense" and not jump_used:
                    alive_three_count += 1
            elif pure == win_length - 3:
                direction_score += score_table['alive2']
        else:  # 死棋型
            if pure >= win_length:
                direction_score += score_table['win5']
            elif pure == win_length - 1:
                # 这里采用“死棋型”的得分
                direction_score += score_table['dead4']
            elif pure == win_length - 2:
                direction_score += score_table['dead3']
            elif pure == win_length - 3:
                direction_score += score_table['dead2']
        
        # 加上跳子 bonus 的得分
        direction_score += bonus * bonus_factor
        
        total_score += direction_score
    
    # 如果是进攻模式，并且有双“真”活三，则额外加分
    if mode == "offense" and alive_three_count >= 2:
        total_score += score_table['double3']
    
    return total_score * factor

def get_opponent(player):
    """
    获取对手棋子标识。
    如果 player 为 1 或 -1，则对手取 -player；
    否则对于 1、2 双方，采用 3 - player。
    """
    if player in [1, -1]:
        return -player
    else:
        return 3 - player

def evaluate_cell(board, x, y, player, win_length=5, offense_weight=1.0, defense_weight=1.0):
    """
    对棋盘上空位 (x,y) 进行评估：
      1. 临时落子 player 计算进攻评分；
      2. 临时落子 opponent 计算防守（对手威胁）评分；
      3. 综合评分 = offense_weight * offense_score + defense_weight * defense_score.
      
    注意：评估过程中均不考虑位置权重，最后由 agent 结合位置偏好调整。
    """
    if board[x, y] != 0:
        raise ValueError("Cell is not empty!")
    original = board[x, y]
    # 进攻评分：假设我方落子
    board[x, y] = player
    offense = compute_score(board, x, y, player, win_length, 'offense')
    board[x, y] = original
    # 防守评分：假设对手落子
    opponent = get_opponent(player)
    board[x, y] = opponent
    defense = compute_score(board, x, y, opponent, win_length, 'defense')
    board[x, y] = original
    combined = offense_weight * offense + defense_weight * defense
    return offense, defense, combined

def evaluate_board_map(board, player, win_length=5, offense_weight=1.0, defense_weight=1.0):
    """
    遍历棋盘所有空位，生成一个评估 map：
    key 为 (x,y) 坐标，value 为字典，包含：
      - 'offense': 进攻评分
      - 'defense': 防守评分
      - 'combined': 加权综合评分
    """
    score_map = {}
    rows, cols = board.shape
    for x in range(rows):
        for y in range(cols):
            if board[x, y] == 0:
                off, defe, comb = evaluate_cell(board, x, y, player, win_length, offense_weight, defense_weight)
                score_map[(x, y)] = {'offense': off, 'defense': defe, 'combined': comb}
    return score_map
