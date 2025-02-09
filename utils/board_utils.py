"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2025-02-09 20:50:26
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2025-02-09 21:17:05
FilePath: /RL-ChessMaster/utils/board_utils.py
Description: 

Copyright (c) 2025 by LiangSong(sl12160010@gmail.com), All Rights Reserved. 
"""
import numpy as np

def check_winner(board, x, y, win_length=5, cache=None):
    """
    检查给定位置是否形成获胜连珠。
    """
    if board.shape[0] < win_length or board.shape[1] < win_length:
        raise ValueError(f"Board size must be at least {win_length}x{win_length}")
    
    player = board[x, y]
    if player == 0:
        return False

    if cache is None:
        cache = {}
    pos_key = (x, y, player, win_length)
    if pos_key in cache:
        return cache[pos_key]

    directions = [
        [(1, 0), (-1, 0)],   # 垂直
        [(0, 1), (0, -1)],   # 水平
        [(1, 1), (-1, -1)],  # 主对角线
        [(1, -1), (-1, 1)]   # 副对角线
    ]
    
    for direction_pair in directions:
        count = 1  # 包含当前位置
        for dx, dy in direction_pair:
            curr_x, curr_y = x + dx, y + dy
            while (0 <= curr_x < board.shape[0] and 
                   0 <= curr_y < board.shape[1] and 
                   board[curr_x, curr_y] == player):
                count += 1
                if count >= win_length:
                    cache[pos_key] = True
                    return True
                curr_x += dx
                curr_y += dy
                
    cache[pos_key] = False
    return False

def is_alive_pattern(board, x, y, length):
    """
    检查是否是活棋型（两端都是空的）。
    这里要求从 (x,y) 出发、连续达到 length 时，两端必须均为空。
    """
    directions = [
        [(1, 0), (-1, 0)],   # 垂直
        [(0, 1), (0, -1)],   # 水平
        [(1, 1), (-1, -1)],  # 主对角线
        [(1, -1), (-1, 1)]   # 副对角线
    ]
    
    player = board[x, y]
    for direction_pair in directions:
        count = 1  # 包括当前位置
        open_ends = 0
        for dx, dy in direction_pair:
            curr_x, curr_y = x, y
            for _ in range(length - 1):
                curr_x += dx
                curr_y += dy
                if (0 <= curr_x < board.shape[0] and 
                    0 <= curr_y < board.shape[1] and 
                    board[curr_x, curr_y] == player):
                    count += 1
                else:
                    break
            if (0 <= curr_x < board.shape[0] and 
                0 <= curr_y < board.shape[1] and 
                board[curr_x, curr_y] == 0):
                open_ends += 1
        if count >= length and open_ends == 2:
            return True
    return False

def analyze_direction(board, x, y, color, directions):
    """
    分析以 (x,y) 为中心、颜色为 color，
    沿 directions 指定的两个方向的情况。
    
    返回：
      - pure_count: 连续同色棋子的数量（包含当前位置）。
      - bonus: 如果检测到“跳子”则额外加分（0.5或1）。
      - total_blocks: 两侧被封堵的数目（边界或对方棋子）。
      - jump_used: 是否检测到跳子。
    """
    def in_bounds(a, b):
        return 0 <= a < board.shape[0] and 0 <= b < board.shape[1]
    
    cont = [0, 0]       # 两个方向的连续同色棋子数量
    jump = [False, False]  # 两个方向是否存在跳子
    block = [0, 0]      # 两个方向是否被封堵

    for i, (dx, dy) in enumerate(directions):
        step = 1
        while True:
            nx = x + dx * step
            ny = y + dy * step
            if not in_bounds(nx, ny):
                block[i] = 1  # 越界视为封堵
                break
            cell = board[nx, ny]
            if cell == color:
                cont[i] += 1
                step += 1
            else:
                if cell != 0:
                    block[i] = 1  # 对方棋子视为封堵
                break
        
        # 若没有连续棋子，尝试判断跳子
        if cont[i] == 0:
            nx = x + dx
            ny = y + dy
            if in_bounds(nx, ny) and board[nx, ny] == 0:
                nx2 = x + dx * 2
                ny2 = y + dy * 2
                if in_bounds(nx2, ny2) and board[nx2, ny2] == color:
                    jump[i] = True

    pure_count = 1 + cont[0] + cont[1]
    total_blocks = block[0] + block[1]
    jump_used = jump[0] or jump[1]
    bonus = 0
    if jump[0] and jump[1]:
        bonus = 1
    elif jump[0] or jump[1]:
        bonus = 0.5 if (cont[0] + cont[1]) > 0 else 1
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

def choose_best_position(positions, board_size):
    """
    从给定的空位列表中选择最靠近中心的点。
    """
    if not positions:
        return None

    # 验证坐标合法性
    for x, y in positions:
        if not (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))):
            raise TypeError("Coordinates must be integers")
        if not (0 <= int(x) < board_size and 0 <= int(y) < board_size):
            raise ValueError("Coordinates must be within board bounds")

    center = board_size // 2
    sorted_positions = sorted(
        positions,
        key=lambda pos: (
            abs(int(pos[0]) - center) + abs(int(pos[1]) - center),
            abs(int(pos[0]) - center),
            abs(int(pos[1]) - center)
        )
    )
    best_x, best_y = sorted_positions[0]
    return (int(best_x), int(best_y))