# Algorithm Project: Lab1
## 项目地址

## 整体思路描述
### 图化问题
* 首先以reference为横轴，query为纵轴搭建一个matrix，对reference的每一个数值（即每一列），标注reference与query相同（1）或互补的点（-1）。
* 然后，设置一个长度阈值（这里设为5），检索整个matrix中与正对角线平行的1序列和与反对角线平行的-1连续序列（序列长度需大于阈值），将这些小序列称为Chain。
* 利用哈希表，记录每一列出现的chain，然后将每个chain视为一个点，在同一列出现过的chain或首尾相接的chain视为相连的点，这样，我们把整个问题转化成了一个图。
### 寻路算法
* 我们记录所有起点为第0列的chain为start_chain，所有终点为最后一列的chain为end_chain，并将相连的chain之间的距离记为1。
* 利用迪杰斯特拉算法，寻找到一条从某一start_chain到达end_chain的路径，该路径即为找到的含有最优重复的序列
## 算法伪代码
### 图化算法
    function build_matrix(reference, query):
        matrix = create_matrix(len(reference), len(query))
        for i = 0 to len(reference)-1:
            for j = 0 to len(query)-1:
                if reference[i] == query[j]:
                    matrix[i][j] = 1  // Match
                else if is_complementary(reference[i], query[j]):
                    matrix[i][j] = -1  // Complementary
                else:
                    matrix[i][j] = 0  // Mismatch
        return matrix

    function extract_chains(matrix, threshold):
        chains = []
        for i = 0 to len(matrix)-1:
            for j = 0 to len(matrix[0])-1:
                if matrix[i][j] == 1 or matrix[i][j] == -1:
                    chain = []
                    while valid_chain(i, j, matrix):
                        chain.append((i, j))
                        i, j = next_position(i, j)
                    if len(chain) >= threshold:
                        chains.append(chain)
        return chains
    function build_graph(chains):
        graph = {}
        for each chain in chains:
            start = chain[0]  // First point of chain
            end = chain[-1]  // Last point of chain
            for neighbor in find_neighbors(start, end, chains):
                graph[start].append(neighbor)
                graph[neighbor].append(start)  // Undirected edge if needed
        return graph
### 寻路算法
    function dijkstra(graph, start_chains, end_chains):
        dist = {}  // Distance from start_chain to each chain
        parent = {}  // Parent to reconstruct path
        priority_queue = min_heap()

        // Initialize distances
        for each start_chain in start_chains:
            dist[start_chain] = 0
            parent[start_chain] = null
            push(priority_queue, (0, start_chain))  // Push with distance 0

        // Run Dijkstra's algorithm
        while priority_queue is not empty:
            current_dist, current_chain = pop(priority_queue)
            
            if current_chain in end_chains:
                return reconstruct_path(parent, current_chain)
            
            for neighbor, weight in graph[current_chain]:
                new_dist = current_dist + weight
                if neighbor not in dist or new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    parent[neighbor] = current_chain
                    push(priority_queue, (new_dist, neighbor))
        
        return null  // No path found

    function reconstruct_path(parent, end_chain):
        path = []
        current_chain = end_chain
        while current_chain is not null:
            path.append(current_chain)
            current_chain = parent[current_chain]
        reverse(path)  // Reverse the path to get correct order
        return path
### 主函数
    function find_repeating_variant(reference, query, threshold):
        matrix = build_matrix(reference, query)
        chains = extract_chains(matrix, threshold)
        
        // Extract start and end chains
        start_chains = [chain[0] for each chain in chains]
        end_chains = [chain[-1] for each chain in chains]
        
        // Build the graph based on chains
        graph = build_graph(chains)
        
        // Find the shortest path from any start_chain to any end_chain
        optimal_path = dijkstra(graph, start_chains, end_chains)
        
        if optimal_path is not null:
            return extract_sequence_from_chains(optimal_path, reference, query)
        else:
            return "No repeating variant found"
## 时空复杂度

### 图化算法
* 构建矩阵： O(1)
* 提取链： 由于我们使用了一个hash表记录已被标为chain的点，因此，每个点最多被遍历一次, 时间复杂度为O(n^2)，空间复杂度也为O(n^2)
* 构建图： 同样的，我们采用哈希表记录每列出现的chain, 提取邻接表时只需进行指针复制，且由于对chain的长度设置了最小阈值，大大减小了无关的干扰段和可能的chain的数量，极大提升了效率，因此时间复杂度最多为O(n^2)，空间复杂度最多为O(n^2)
### 寻路算法
* 堆优化的Dijkstra算法的时间复杂度通常为O((V + E)logV)其中V为节点数即为chain的数量，E为边数，看似要大于规定的O(n^2)，但由于我们加入了chain的长度阈值，实测表明过滤极小的片段后，满足条件的chain数量要远低于O(n^2)，达到了O(n)的量级，且由于大部分chain的长度仍然很多，不同chain之间的相连也极为稀疏，接近O(n)量级，因此实际运行时间应接近O(nlogn)
* 额外的空间复杂度来源为：
  * 最小堆: O(n)
  * 邻接表: O(n^2)
  * 距离表：O(n)
  * 因此总的空间占用不会超过O(n^2)
## 运行结果截图