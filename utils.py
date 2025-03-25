import numpy as np
import matplotlib.pyplot as plt


# 定义碱基互补关系
complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

class Chain():
    def __init__(self, start, end, len):
        self.start = start
        self.end = end
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

def get_chains(matrix):
    pos_chains = []
    neg_chains = []
    cols_hash = []
    for i in range(matrix.shape[1]):
        col_hash = []
        
    

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
