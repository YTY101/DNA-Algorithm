import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import heapq

# 定义碱基互补关系
complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

class Chain():
    def __init__(self, startX, startY, endX, endY, len, type):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY
        self.len = len
        self.type = type

class ProirityQueue():
    def __init__(self):
        self.heap = []

    def _compare(self, item1, item2):
        return item1 < item2

    def _shift_up(self, i):
        parent = (i - 1) // 2
        if parent >= 0 and self._compare(self.heap[i], self.heap[parent]):
            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
            self._shift_up(parent)

    def _shift_down(self, i):
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < len(self.heap) and self._compare(self.heap[left], self.heap[smallest]):
            smallest = left
        if right < len(self.heap) and self._compare(self.heap[right], self.heap[smallest]):
            smallest = right
        if smallest!= i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self._shift_down(smallest)

    def push(self, item):
        self.heap.append(item)
        self._shift_up(len(self.heap) - 1)
        
    def pop(self):
        if len(self.heap) == 0:
            return None
        item = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self._shift_down(0)
        return item
    
    def is_empty(self):
        return len(self.heap) == 0


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
                chain = Chain(startX, startY, endX, endY, len, 'P')
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
                chain = Chain(startX, startY, endX, endY, len, 'N')
                if threshold == -1 or len >= threshold:
                    neg_chains.append(chain)
            
    return pos_chains, neg_chains

def _get_neigbours(chains, col_start_chains, end_chains):
    neighbours = {}
    for i in range(len(chains)):
        neighbours[i] = []
        if i not in end_chains:
            for cover_col in range(chains[i].startX + 1, chains[i].endX + 2):
                for chain_id in col_start_chains[cover_col]:
                    neighbours[i].append(chain_id)
    return neighbours

def get_graph(pos_chains, neg_chains, matrix):
    chains = pos_chains + neg_chains
    chains_num = len(chains)
    
    start_chains = []
    end_chains = []

    col_start_chains = {}
    for i in range(matrix.shape[1]):
        col_start_chains[i] = []
    
    for i in range(chains_num):
        if chains[i].startX == 0:
            start_chains.append(i)
            
        if chains[i].endX + 1 == matrix.shape[1]:
            end_chains.append(i)

        col_start_chains[chains[i].startX].append(i)

    # print("Col_Start_Chain: ", col_start_chains)
    # print("End_Chains", end_chains)
    neighbours = _get_neigbours(chains, col_start_chains, end_chains)
    return  chains, start_chains, end_chains, neighbours



def find_path(chains, start_chains, end_chains, neighbours):
    distances = {}
    pq = []

    for i in range(len(chains)):
        distances[i] = -1
    for i in start_chains:
        distances[i] = 0
        heapq.heappush(pq, (distances[i], i))  
    
    while pq:
        current_distance, current_chain = heapq.heappop(pq)
        # print("Current Chain: ", current_chain, "Current Distance: ", current_distance)
        if current_distance != -1 and current_distance > distances[current_chain]:
            continue
        for neighbour in neighbours[current_chain]:
            distance = current_distance + 1
            if distances[neighbour] == -1 or distance < distances[neighbour]:
                distances[neighbour] = distance
                heapq.heappush(pq, (distance, neighbour))
            if neighbour in end_chains:
                # print("Hit the end: ", neighbour)
                return distances[neighbour]
        # print("Distances: ", distances)
    return -1
def show_data(matrix):
    fig, ax = plt.subplots(figsize=(10, 10))

    cax = ax.imshow(matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

    ax.invert_yaxis()
    
    ax.set_xlabel('query')
    ax.set_ylabel('reference')

    ax.set_title('Query vs Reference Sequence Alignment')

    plt.tight_layout()
    plt.show()

def show_chains(pos_chains, neg_chains):
    fig, ax = plt.subplots(figsize=(10, 10))

    for chain in pos_chains:
        ax.plot([chain.startX, chain.endX], [chain.startY, chain.endY], color='blue')

    for chain in neg_chains:
        ax.plot([chain.startX, chain.endX], [chain.startY, chain.endY], color='red')

    
    ax.set_xlabel('query')
    ax.set_ylabel('reference')

    ax.set_title('Query vs Reference Sequence Alignment')

    plt.tight_layout()
    plt.show()
    
def show_chains_effective(pos_chains, neg_chains):
    fig, ax = plt.subplots(figsize=(10, 10))

    pos_lines = []
    neg_lines = []

    for chain in pos_chains:
        pos_lines.append([[chain.startX, chain.startY], [chain.endX, chain.endY]])

    for chain in neg_chains:
        neg_lines.append([[chain.startX, chain.startY], [chain.endX, chain.endY]])

    pos_line_collection = LineCollection(pos_lines, colors='blue', linewidths=1)
    neg_line_collection = LineCollection(neg_lines, colors='red', linewidths=1)

    ax.add_collection(pos_line_collection)
    ax.add_collection(neg_line_collection)

    ax.set_xlim(min(chain.startX for chain in pos_chains + neg_chains) - 10, 
                max(chain.endX for chain in pos_chains + neg_chains) + 10)
    ax.set_ylim(min(chain.startY for chain in pos_chains + neg_chains) - 10, 
                max(chain.endY for chain in pos_chains + neg_chains) + 10)

    ax.set_xlabel('query')
    ax.set_ylabel('reference')

    ax.set_title('Query vs Reference Sequence Alignment')

    plt.tight_layout()
    plt.show()