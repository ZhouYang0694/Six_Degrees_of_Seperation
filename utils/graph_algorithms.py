"""
图算法模块
包含计算最短路径等图相关算法
"""
from collections import deque
from typing import List, Tuple, Dict, Optional
import numpy as np


def bfs_shortest_path(edges: List[Tuple[int, int]], start: int, end: int, n: int) -> int:
    """
    使用BFS算法计算两个节点之间的最短距离
    
    Args:
        edges: 边列表
        start: 起始节点
        end: 终止节点
        n: 总节点数
        
    Returns:
        int: 最短距离，如果不连通返回-1
    """
    if start == end:
        return 0
    
    # 构建邻接列表
    adj_list = [[] for _ in range(n)]
    for edge in edges:
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])
    
    # BFS搜索
    queue = deque([(start, 0)])  # (节点, 距离)
    visited = set([start])
    
    while queue:
        current_node, distance = queue.popleft()
        
        for neighbor in adj_list[current_node]:
            if neighbor == end:
                return distance + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1  # 不连通


def multi_source_bfs(adj_list: List[List[int]], source: int, n: int) -> np.ndarray:
    """
    从单个源点开始的多目标BFS，计算到所有其他节点的最短距离
    比多次单独BFS更高效
    
    Args:
        adj_list: 邻接列表
        source: 源节点
        n: 总节点数
        
    Returns:
        np.ndarray: 距离数组，-1表示不连通
    """
    distances = np.full(n, -1, dtype=int)
    distances[source] = 0
    
    queue = deque([source])
    
    while queue:
        current = queue.popleft()
        current_dist = distances[current]
        
        for neighbor in adj_list[current]:
            if distances[neighbor] == -1:  # 未访问
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    return distances


def optimized_all_pairs_shortest_path(edges: List[Tuple[int, int]], n: int) -> np.ndarray:
    """
    优化的全对最短路径算法
    对于稀疏图使用多源BFS，对于密集图使用Floyd-Warshall
    
    Args:
        edges: 边列表
        n: 总节点数
        
    Returns:
        np.ndarray: 距离矩阵，-1表示不连通
    """
    # 构建邻接列表
    adj_list = [[] for _ in range(n)]
    for edge in edges:
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])
    
    # 计算图的密度
    max_edges = n * (n - 1) // 2
    density = len(edges) / max_edges if max_edges > 0 else 0
    
    # 对于稀疏图（密度 < 0.1），使用多源BFS更高效
    if density < 0.1:
        print(f"Using multi-source BFS for sparse graph (density: {density:.3f})")
        distance_matrix = np.full((n, n), -1, dtype=int)
        
        # 对每个节点运行BFS
        for i in range(n):
            distances = multi_source_bfs(adj_list, i, n)
            distance_matrix[i] = distances
        
        return distance_matrix
    else:
        print(f"Using Floyd-Warshall for dense graph (density: {density:.3f})")
        return floyd_warshall_shortest_paths(edges, n)


def floyd_warshall_shortest_paths(edges: List[Tuple[int, int]], n: int) -> np.ndarray:
    """
    使用Floyd-Warshall算法计算所有节点对之间的最短距离
    适用于需要计算大量节点对最短距离的情况
    
    Args:
        edges: 边列表
        n: 总节点数
        
    Returns:
        np.ndarray: 距离矩阵，-1表示不连通
    """
    # 初始化距离矩阵
    dist = np.full((n, n), float('inf'))
    
    # 对角线设为0
    for i in range(n):
        dist[i][i] = 0
    
    # 设置直接连接的距离为1
    for edge in edges:
        dist[edge[0]][edge[1]] = 1
        dist[edge[1]][edge[0]] = 1
    
    # Floyd-Warshall核心算法
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # 将无穷大值转换为-1
    dist[dist == float('inf')] = -1
    
    return dist.astype(int)


def calculate_reachability_stats(distance_matrix: np.ndarray, max_degree: int = 6) -> Dict:
    """
    计算可达性统计信息
    
    Args:
        distance_matrix: 距离矩阵
        max_degree: 最大度数
        
    Returns:
        Dict: 统计信息字典
    """
    n = distance_matrix.shape[0]
    stats = {}
    
    # 计算每个度数下的可达性
    for degree in range(1, max_degree + 1):
        # 1. 计算能够在degree步内到达所有其他人的节点数量
        # （即对于某个人，他能在degree步内到达所有其他人）
        universal_reachable_nodes = 0
        
        for i in range(n):
            # 检查节点i是否能在degree步内到达所有其他节点
            can_reach_all = True
            for j in range(n):
                if i != j and (distance_matrix[i][j] == -1 or distance_matrix[i][j] > degree):
                    can_reach_all = False
                    break
            if can_reach_all:
                universal_reachable_nodes += 1
        
        # 2. 计算距离≤degree的节点对数量
        reachable_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if 0 < distance_matrix[i][j] <= degree:
                    reachable_pairs += 1
        
        total_pairs = n * (n - 1) // 2
        pair_coverage = reachable_pairs / total_pairs if total_pairs > 0 else 0
        universal_coverage = universal_reachable_nodes / n if n > 0 else 0
        
        stats[degree] = {
            'reachable_pairs': reachable_pairs,
            'total_pairs': total_pairs,
            'pair_coverage': pair_coverage,
            'universal_reachable_nodes': universal_reachable_nodes,
            'total_nodes': n,
            'universal_coverage': universal_coverage
        }
    
    return stats


def analyze_network_connectivity(edges: List[Tuple[int, int]], n: int) -> Dict:
    """
    分析网络的连通性
    
    Args:
        edges: 边列表
        n: 总节点数
        
    Returns:
        Dict: 连通性分析结果
    """
    # 构建邻接列表
    adj_list = [[] for _ in range(n)]
    for edge in edges:
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])
    
    # 找出所有连通分量
    visited = [False] * n
    components = []
    
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)
    
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    # 计算最大连通分量
    largest_component = max(components, key=len) if components else []
    
    return {
        'num_components': len(components),
        'largest_component_size': len(largest_component),
        'connectivity_ratio': len(largest_component) / n if n > 0 else 0,
        'components': components
    }


def get_diameter(distance_matrix: np.ndarray) -> int:
    """
    计算网络的直径（最大最短路径距离）
    
    Args:
        distance_matrix: 距离矩阵
        
    Returns:
        int: 网络直径
    """
    # 只考虑连通的节点对
    valid_distances = distance_matrix[distance_matrix > 0]
    return int(np.max(valid_distances)) if len(valid_distances) > 0 else 0


def get_average_path_length(distance_matrix: np.ndarray) -> float:
    """
    计算平均路径长度
    
    Args:
        distance_matrix: 距离矩阵
        
    Returns:
        float: 平均路径长度
    """
    # 只考虑连通的节点对
    valid_distances = distance_matrix[distance_matrix > 0]
    return float(np.mean(valid_distances)) if len(valid_distances) > 0 else 0.0
