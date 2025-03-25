import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# 定义碱基互补关系
complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

class Chain():
    def __init__(self, startX, startY, endX, endY, len):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY
        self.len = len

def load_data(reference, query):
    # 创建一个二维矩阵来表示匹配情况
    # 如果 query 和 reference 在某个位置匹配，值为 1，如果互补，值为 -1，否则为 0
    matrix = np.zeros((len(reference), len(query)))

    for i, ref_base in enumerate(reference):
        for j, query_base in enumerate(query):
            if ref_base == query_base:
                matrix[i, j] = 1  # 如果相同，设置为 1
            elif complement.get(query_base) == ref_base:
                matrix[i, j] = -1  # 如果互补，设置为 -1
            else:
                matrix[i, j] = 0  # 否则，设置为 0
    return  matrix

def get_chains(matrix, threshold=-1):
    pos_chains = []
    neg_chains = []
    marked_hash = {}
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            marked_hash[(i, j)] = 0
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            i_now = i
            j_now = j
            startX = i_now
            startY = j_now
            if matrix[j, i] == 1:
                while(i_now < matrix.shape[1] and j_now < matrix.shape[0] and marked_hash[(i_now, j_now)] != 1 and matrix[j_now, i_now] == 1):
                    marked_hash[(i_now, j_now)] = 1
                    i_now += 1
                    j_now += 1
                endX = i_now - 1
                endY = j_now - 1
                len = endX - startX + 1
                chain = Chain(startX, startY, endX, endY, len)
                if threshold == -1 or len >= threshold:
                    pos_chains.append(chain)    
                
            if matrix[j, i] == -1:
                while(i_now < matrix.shape[1] and j_now >= 0 and marked_hash[(i_now, j_now)] != 1 and matrix[j_now, i_now] == -1):
                    marked_hash[(i_now, j_now)] = 1
                    i_now += 1
                    j_now -= 1
                endX = i_now - 1
                endY = j_now + 1
                len = endX - startX + 1
                chain = Chain(startX, startY, endX, endY, len)
                if threshold == -1 or len >= threshold:
                    neg_chains.append(chain)
            
    return pos_chains, neg_chains
                

def show_data(matrix):
    # 创建一个图形
    fig, ax = plt.subplots(figsize=(10, 10))

    # 使用imshow绘制矩阵
    cax = ax.imshow(matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

    # 反转y轴显示方向
    ax.invert_yaxis()
    
    ax.set_xlabel('query')
    ax.set_ylabel('reference')

    # # 添加颜色条
    # fig.colorbar(cax)

    # 设置标题
    ax.set_title('Query vs Reference Sequence Alignment')

    # 显示图形
    plt.tight_layout()
    plt.show()

def show_chains(pos_chains, neg_chains):
    # 创建一个图形
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制匹配链
    for chain in pos_chains:
        ax.plot([chain.startX, chain.endX], [chain.startY, chain.endY], color='blue')

    # 绘制互补链
    for chain in neg_chains:
        ax.plot([chain.startX, chain.endX], [chain.startY, chain.endY], color='red')

    # 反转y轴显示方向
    # ax.invert_yaxis()
    
    ax.set_xlabel('query')
    ax.set_ylabel('reference')

    # 设置标题
    ax.set_title('Query vs Reference Sequence Alignment')

    # 显示图形
    plt.tight_layout()
    plt.show()
    
def show_chains_effective(pos_chains, neg_chains):
    # 创建一个图形
    fig, ax = plt.subplots(figsize=(10, 10))

    # 准备绘制的数据：正链和负链
    pos_lines = []
    neg_lines = []

    # 将正链的线段添加到 pos_lines 中
    for chain in pos_chains:
        pos_lines.append([[chain.startX, chain.startY], [chain.endX, chain.endY]])

    # 将负链的线段添加到 neg_lines 中
    for chain in neg_chains:
        neg_lines.append([[chain.startX, chain.startY], [chain.endX, chain.endY]])

    # 创建 LineCollection 来批量绘制线段
    pos_line_collection = LineCollection(pos_lines, colors='blue', linewidths=1)
    neg_line_collection = LineCollection(neg_lines, colors='red', linewidths=1)

    # 将线段添加到图形中
    ax.add_collection(pos_line_collection)
    ax.add_collection(neg_line_collection)

    # 设置坐标轴范围
    ax.set_xlim(min(chain.startX for chain in pos_chains + neg_chains) - 10, 
                max(chain.endX for chain in pos_chains + neg_chains) + 10)
    ax.set_ylim(min(chain.startY for chain in pos_chains + neg_chains) - 10, 
                max(chain.endY for chain in pos_chains + neg_chains) + 10)

    # 设置坐标轴标签
    ax.set_xlabel('query')
    ax.set_ylabel('reference')

    # 设置标题
    ax.set_title('Query vs Reference Sequence Alignment')

    # 显示图形
    plt.tight_layout()
    plt.show()