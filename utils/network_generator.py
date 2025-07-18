"""
人际关系网络生成器
用于生成不同类型的人际关系网络
"""
import random
import numpy as np
from typing import List, Tuple, Set


class NetworkGenerator:
    """
    人际关系网络生成器类
    支持多种网络生成算法，为后续扩展提供基础
    """
    
    def __init__(self, n: int, k: int, seed: int = None):
        """
        初始化网络生成器
        
        Args:
            n: 人数（节点数）
            k: 每个人的平均连接数
            seed: 随机种子，用于结果复现
        """
        self.n = n
        self.k = k
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_network(self) -> List[Tuple[int, int]]:
        """
        生成随机网络（Erdős–Rényi模型）
        每个人随机与k个其他人建立连接
        
        Returns:
            List[Tuple[int, int]]: 边列表，每个元组表示一条边
        """
        edges = set()
        
        # 为每个人生成k个随机连接
        for person in range(self.n):
            # 获取可能的连接对象（除了自己）
            possible_connections = list(range(self.n))
            possible_connections.remove(person)
            
            # 随机选择k个人建立连接
            connections = random.sample(possible_connections, 
                                      min(self.k, len(possible_connections)))
            
            for connected_person in connections:
                # 确保边是无向的（较小的节点在前）
                edge = (min(person, connected_person), 
                       max(person, connected_person))
                edges.add(edge)
        
        return list(edges)
    
    def generate_small_world_network(self, p: float = 0.1) -> List[Tuple[int, int]]:
        """
        生成小世界网络（Watts-Strogatz模型）
        先构建规则网络，然后随机重连一些边
        
        Args:
            p: 重连概率
            
        Returns:
            List[Tuple[int, int]]: 边列表
        """
        edges = set()
        
        # 首先构建规则网络：每个节点与其最近的k个邻居相连
        for i in range(self.n):
            for j in range(1, self.k // 2 + 1):
                # 右侧邻居
                neighbor = (i + j) % self.n
                edge = (min(i, neighbor), max(i, neighbor))
                edges.add(edge)
                
                # 左侧邻居
                neighbor = (i - j) % self.n
                edge = (min(i, neighbor), max(i, neighbor))
                edges.add(edge)
        
        # 随机重连边
        edges_list = list(edges)
        rewired_edges = set()
        
        for edge in edges_list:
            if random.random() < p:
                # 重连这条边
                node1, node2 = edge
                # 为node1选择一个新的随机邻居
                possible_neighbors = list(range(self.n))
                possible_neighbors.remove(node1)
                # 移除已经连接的邻居
                current_neighbors = {node2}
                for e in rewired_edges:
                    if e[0] == node1:
                        current_neighbors.add(e[1])
                    elif e[1] == node1:
                        current_neighbors.add(e[0])
                
                available_neighbors = [n for n in possible_neighbors 
                                     if n not in current_neighbors]
                
                if available_neighbors:
                    new_neighbor = random.choice(available_neighbors)
                    new_edge = (min(node1, new_neighbor), 
                               max(node1, new_neighbor))
                    rewired_edges.add(new_edge)
                else:
                    rewired_edges.add(edge)
            else:
                rewired_edges.add(edge)
        
        return list(rewired_edges)
    
    def generate_scale_free_network(self) -> List[Tuple[int, int]]:
        """
        生成无标度网络（Barabási-Albert模型）
        新加入的节点优先连接到度数高的节点
        
        Returns:
            List[Tuple[int, int]]: 边列表
        """
        edges = []
        degrees = [0] * self.n
        
        # 从一个小的完全图开始
        initial_nodes = min(self.k, self.n)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                edges.append((i, j))
                degrees[i] += 1
                degrees[j] += 1
        
        # 逐个添加剩余节点
        for new_node in range(initial_nodes, self.n):
            # 计算每个现有节点被选中的概率（与度数成正比）
            total_degree = sum(degrees[:new_node])
            if total_degree == 0:
                continue
                
            probabilities = [degrees[i] / total_degree for i in range(new_node)]
            
            # 选择要连接的节点
            connections_to_make = min(self.k, new_node)
            connected_nodes = set()
            
            while len(connected_nodes) < connections_to_make:
                # 根据度数概率选择节点
                selected_node = np.random.choice(new_node, p=probabilities)
                connected_nodes.add(selected_node)
            
            # 建立连接
            for connected_node in connected_nodes:
                edges.append((connected_node, new_node))
                degrees[connected_node] += 1
                degrees[new_node] += 1
        
        return edges
    
    def edges_to_adjacency_matrix(self, edges: List[Tuple[int, int]]) -> np.ndarray:
        """
        将边列表转换为邻接矩阵
        
        Args:
            edges: 边列表
            
        Returns:
            np.ndarray: 邻接矩阵
        """
        adj_matrix = np.zeros((self.n, self.n), dtype=bool)
        
        for edge in edges:
            i, j = edge
            adj_matrix[i][j] = True
            adj_matrix[j][i] = True
        
        return adj_matrix
    
    def get_network_stats(self, edges: List[Tuple[int, int]]) -> dict:
        """
        获取网络统计信息
        
        Args:
            edges: 边列表
            
        Returns:
            dict: 包含网络统计信息的字典
        """
        # 计算度分布
        degrees = [0] * self.n
        for edge in edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        
        stats = {
            'total_edges': len(edges),
            'average_degree': sum(degrees) / self.n,
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'degree_distribution': degrees
        }
        
        return stats
