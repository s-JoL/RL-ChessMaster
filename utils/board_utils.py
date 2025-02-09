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
    """检查给定位置是否形成获胜连珠
    
    Args:
        board (numpy.ndarray): 15x15的棋盘矩阵, 0表示空位, 1和2表示玩家棋子
        x (int): 待检查位置的行索引 (0-14)
        y (int): 待检查位置的列索引 (0-14)
        win_length (int, optional): 获胜所需连珠长度, 默认为5
        cache (dict, optional): 缓存已检查过的位置和结果
        
    Returns:
        bool: 如果形成指定长度的连续棋子则返回True, 否则返回False
    """
    if board.shape[0] < win_length or board.shape[1] < win_length:
        raise ValueError(f"Board size must be at least {win_length}x{win_length}")
    
    player = board[x, y]
    if player == 0:
        return False
        
    if cache is None:
        cache = {}
    # 缓存 key 中增加 win_length，避免不同判断条件混淆
    pos_key = (x, y, player, win_length)
    if pos_key in cache:
        return cache[pos_key]

    # 定义四个检查方向及其反方向
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
    """检查是否是活棋型（两端都是空的）
    
    这里要求从 (x,y) 出发、连续达到 length 时，
    两端必须均为空（即 open_ends == 2）。
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
            # 检查端点是否为空
            if (0 <= curr_x < board.shape[0] and 
                0 <= curr_y < board.shape[1] and 
                board[curr_x, curr_y] == 0):
                open_ends += 1
        # 活棋型要求两端都为空
        if count >= length and open_ends == 2:
            return True
    return False

def analyze_direction(board, x, y, color, directions):
    """
    分析在棋盘 board 上，以 (x, y) 为中心、颜色为 color，
    沿 directions 指定的两个方向的情况。
    
    返回四个值：
      - pure_count: 基于连续子计算的基础数值（计入候选点自身）。
      - bonus: 如果检测到“跳子”则额外加分（可能为 0.5 或 1）。
      - total_blocks: 两侧被封堵的数目（边界或对方棋子）。
      - jump_used: 布尔值，表示是否有跳子的情况。
    """
    def in_bounds(a, b):
        return 0 <= a < board.shape[0] and 0 <= b < board.shape[1]
    
    cont = [0, 0]      # 两个方向上连续同色棋子数量
    jump = [False, False]  # 两个方向上是否检测到跳子
    block = [0, 0]     # 两个方向上是否被封堵

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
        
        # 如果这一方向没有连续子，则尝试判断是否存在“跳子”
        if cont[i] == 0:
            nx = x + dx
            ny = y + dy
            if in_bounds(nx, ny) and board[nx, ny] == 0:
                # 检查隔一格的位置
                nx2 = x + dx * 2
                ny2 = y + dy * 2
                if in_bounds(nx2, ny2) and board[nx2, ny2] == color:
                    jump[i] = True

    pure_count = 1 + cont[0] + cont[1]  # 基础计数：包括当前位置及连续子
    total_blocks = block[0] + block[1]
    jump_used = jump[0] or jump[1]
    bonus = 0
    if jump[0] and jump[1]:
        bonus = 1
    elif jump[0] or jump[1]:
        # 如果已有连续子，则跳子威力减半，否则全算1分
        if (cont[0] + cont[1]) > 0:
            bonus = 0.5
        else:
            bonus = 1
    return pure_count, bonus, total_blocks, jump_used

def evaluate_position(board, x, y, player, win_length=5):
    """
    评估候选落子位置 (x,y) 的潜在价值，同时返回该位置扩展后在某个方向上的最大连子数。
    
    返回一个元组：(位置评分, 最大有效连子数, 最佳方向)
    
    其中“最大有效连子数” = pure_count + bonus 。（注：在判断棋型时，我们只依据 pure_count，
    bonus 仅作为附加得分，不参与是否能成五的判断。）
    """
    directions = [
        [(0, 1), (0, -1)],    # 水平
        [(1, 0), (-1, 0)],    # 垂直
        [(1, 1), (-1, -1)],   # 右下对角线
        [(1, -1), (-1, 1)]    # 左下对角线
    ]
    
    total_score = 0
    max_effective = 0
    best_direction = None
    # 判断对手，支持 1/2 或 1/(-1) 的表示方式
    if player in [1, -1]:
        opponent = -player
    else:
        opponent = 1 if player == 2 else 2

    # 基础分值表
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
    bonus_factor = 100  # 跳子 bonus 额外加分因子

    alive_three_count = 0  # 用于统计“真”活三（没有跳子辅助）的数量

    for direction_pair in directions:
        # 我方评估
        pure, bonus, blocks, jump_used = analyze_direction(board, x, y, player, direction_pair)
        effective_count = pure + bonus  # 供显示参考，但分类时以 pure 为准

        # 更新最佳方向（根据有效连子数）
        if effective_count > max_effective:
            max_effective = effective_count
            best_direction = direction_pair

        direction_score = 0

        # 我方棋形评分（严格依据 pure 进行分类）
        if blocks == 0:  # 活棋型
            if pure >= win_length:
                direction_score += score_table['win5']
            elif pure == win_length - 1:
                direction_score += score_table['alive4']
            elif pure == win_length - 2:
                direction_score += score_table['alive3']
                if not jump_used:  # 没有使用跳子，认为是真活三
                    alive_three_count += 1
            elif pure == win_length - 3:
                direction_score += score_table['alive2']
        else:  # 死棋型
            if pure >= win_length:
                direction_score += score_table['win5']
            elif pure == win_length - 1:
                direction_score += score_table['dead4']
            elif pure == win_length - 2:
                direction_score += score_table['dead3']
            elif pure == win_length - 3:
                direction_score += score_table['dead2']
        # 加上跳子 bonus 额外得分（不影响棋型判断）
        direction_score += bonus * bonus_factor

        # 对手棋型评分（防守考虑，仅计算活型时对手威胁）
        pure_op, bonus_op, blocks_op, jump_used_op = analyze_direction(board, x, y, opponent, direction_pair)
        if blocks_op == 0:
            if pure_op >= win_length - 1:
                direction_score += score_table['win5'] * 0.8
            elif pure_op == win_length - 2:
                direction_score += score_table['alive4'] * 0.8
            elif pure_op == win_length - 3:
                direction_score += score_table['alive3'] * 0.8
            # 同时考虑对手跳子 bonus
            direction_score += bonus_op * bonus_factor * 0.8

        total_score += direction_score

    # 如果存在双“真”活三则额外加分
    if alive_three_count >= 2:
        total_score += score_table['double3']

    # 加上位置权重
    position_weight = calculate_position_weight(board.shape[0], x, y)
    total_score = int(total_score * position_weight)
    
    return total_score, max_effective, best_direction

def choose_best_position(positions, board_size):
    """从多个可选点中选择最接近中心的点

    Args:
        positions: 可选位置列表，每个元素为 (x, y) 坐标元组
        board_size: 棋盘大小

    Returns:
        tuple: 最佳位置 (x, y)，如果 positions 为空则返回 None

    Raises:
        TypeError: 如果坐标包含非整数值
        ValueError: 如果坐标超出棋盘范围或为负
    """
    if not positions:
        return None

    # 验证输入：同时接受 Python 内置 int 和 numpy 的整数类型
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

def calculate_position_weight(board_size, x, y):
    """计算位置权重，靠近中心的位置权重更高"""
    center = board_size // 2
    distance = abs(x - center) + abs(y - center)
    max_distance = 2 * center
    weight = 1 - (distance / max_distance) * 0.5
    return weight

def find_best_block(board, opponent):
    """
    改进版：扫描棋盘，检测对手所有连续棋子链，
    然后选择最长链的两端中一个空位作为堵截点，
    如果有多个候选，则优先选择靠近中心的落点。
    """
    rows, cols = board.shape
    # 定义四个方向，每个方向用一对相反的单位向量表示
    directions = [
        ((1, 0), (-1, 0)),    # 垂直
        ((0, 1), (0, -1)),    # 水平
        ((1, 1), (-1, -1)),   # 主对角线
        ((1, -1), (-1, 1))    # 副对角线
    ]
    chains = []
    # 遍历棋盘，找到对手棋子，并检测从该点开始的连续链
    for x in range(rows):
        for y in range(cols):
            if board[x, y] == opponent:
                for d in directions:
                    # 为避免重复统计，只在“链的起点”进行统计：
                    # 如果在 d[0] 方向的反方向上有同色棋子，则略过
                    dx, dy = d[0]
                    prev_x, prev_y = x - dx, y - dy
                    if 0 <= prev_x < rows and 0 <= prev_y < cols and board[prev_x, prev_y] == opponent:
                        continue
                    # 从 (x,y) 沿 d[0] 方向统计连续同色棋子
                    chain_length = 0
                    cx, cy = x, y
                    while 0 <= cx < rows and 0 <= cy < cols and board[cx, cy] == opponent:
                        chain_length += 1
                        cx += dx
                        cy += dy
                    # 此时，(cx, cy) 就是链的结束后第一个非对手棋子的坐标（可能越界或空位）
                    # 记录两个端点：链的起点反方向的空位和链的末端空位
                    endpoint1 = (x - dx, y - dy)
                    endpoint2 = (cx, cy)
                    chains.append((chain_length, endpoint1, endpoint2, (dx, dy)))
    # 如果没有检测到任何链，则返回 None
    if not chains:
        return None

    # 选出最长的链（如果有多个，则后续候选中只考虑最长链）
    max_length = max(chain[0] for chain in chains)
    # 如果最长链长度小于 2（例如只有单个棋子），则不认为构成严重威胁
    if max_length < 2:
        return None

    # 从所有最长链中，收集端点中空着的位置
    candidates = []
    for chain in chains:
        if chain[0] == max_length:
            for endpoint in (chain[1], chain[2]):
                ex, ey = endpoint
                if 0 <= ex < rows and 0 <= ey < cols and board[ex, ey] == 0:
                    candidates.append(endpoint)
    if not candidates:
        return None

    # 根据到棋盘中心的曼哈顿距离排序，距离近者优先
    center = rows // 2
    candidates.sort(key=lambda pos: abs(pos[0]-center) + abs(pos[1]-center))
    return candidates[0]

def find_best_extend(board, player, win_length=5):
    """查找最佳扩展点，优先形成有价值的棋型
    
    Args:
        board: 当前棋盘状态
        player: 当前玩家标识
        win_length: 获胜所需连子数
        
    Returns:
        tuple: 最佳扩展位置 (x, y)，如果没有合适位置返回 None
    """
    best_positions = []
    max_score = -float('inf')
    
    # 遍历所有空位，计算每个位置的评估得分
    for x, y in zip(*np.where(board == 0)):
        board[x, y] = player
        score, effective_count, best_direction = evaluate_position(board, x, y, player, win_length)
        if score > max_score:
            max_score = score
            best_positions = [(x, y)]
        elif score == max_score:
            best_positions.append((x, y))
        board[x, y] = 0  # 恢复棋盘
    
    # 增加阈值判断，如果最高评分过低，则认为没有有效扩展机会，返回 None
    if max_score <= 1:
        return None
    
    return choose_best_position(best_positions, board.shape[0])

# 示例调用（调试用）：
if __name__ == "__main__":
    # 创建一个空棋盘
    board = np.zeros((15, 15), dtype=int)
    # 假设1为我方，2为对手
    # 模拟几个棋子落点
    board[7, 7] = 1
    board[7, 8] = 1
    board[7, 9] = 1
    board[8, 7] = 2
    board[9, 7] = 2
    
    print("最佳堵截点（针对对手）:", find_best_block(board, opponent=2))
    print("最佳扩展点（我方）:", find_best_extend(board, player=1))
