"""
简单演示版本 - 用于快速测试六度分隔理论
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用内置模块避免依赖问题
from collections import deque
import random
import math

class SimpleNetworkGenerator:
    """简化的网络生成器"""
    
    def __init__(self, n, k, seed=None):
        self.n = n
        self.k = k
        if seed:
            random.seed(seed)
    
    def generate_random_network(self):
        """生成随机网络"""
        edges = set()
        for i in range(self.n):
            # 为每个节点随机选择k个连接
            available = list(range(self.n))
            available.remove(i)
            connections = random.sample(available, min(self.k, len(available)))
            
            for j in connections:
                edge = (min(i, j), max(i, j))
                edges.add(edge)
        
        return list(edges)

def simple_bfs_all_pairs(edges, n):
    """简化的全对最短路径计算"""
    # 构建邻接列表
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # 距离矩阵
    distances = [[-1] * n for _ in range(n)]
    
    # 对每个节点运行BFS
    for start in range(n):
        distances[start][start] = 0
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if distances[start][neighbor] == -1:
                    distances[start][neighbor] = distances[start][current] + 1
                    queue.append(neighbor)
    
    return distances

def analyze_reachability(distances, n):
    """分析可达性"""
    stats = {}
    
    for degree in range(1, 7):
        # 计算能在degree步内到达所有人的节点数
        universal_count = 0
        for i in range(n):
            can_reach_all = True
            for j in range(n):
                if i != j and (distances[i][j] == -1 or distances[i][j] > degree):
                    can_reach_all = False
                    break
            if can_reach_all:
                universal_count += 1
        
        # 计算距离≤degree的节点对数
        reachable_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if 0 < distances[i][j] <= degree:
                    reachable_pairs += 1
        
        total_pairs = n * (n - 1) // 2
        
        stats[degree] = {
            'universal_count': universal_count,
            'universal_ratio': universal_count / n,
            'reachable_pairs': reachable_pairs,
            'pair_ratio': reachable_pairs / total_pairs if total_pairs > 0 else 0
        }
    
    return stats

def simple_demo():
    """简单演示"""
    print("Simple Six Degrees of Separation Demo")
    print("="*40)
    
    N = 50  # 较小的网络用于演示
    k = 6
    
    print(f"Generating random network: N={N}, k={k}")
    
    # 生成网络
    gen = SimpleNetworkGenerator(N, k, seed=42)
    edges = gen.generate_random_network()
    
    print(f"Generated {len(edges)} edges")
    
    # 计算最短路径
    print("Calculating shortest paths...")
    distances = simple_bfs_all_pairs(edges, N)
    
    # 分析可达性
    stats = analyze_reachability(distances, N)
    
    print("\nResults:")
    print("-" * 30)
    
    for degree in range(1, 7):
        s = stats[degree]
        print(f"Within {degree} degree(s):")
        print(f"  Universal reachability: {s['universal_ratio']:.1%} "
              f"({s['universal_count']}/{N} people)")
        print(f"  Pair connectivity: {s['pair_ratio']:.1%} "
              f"({s['reachable_pairs']}/{N*(N-1)//2} pairs)")
    
    # 计算网络直径
    max_dist = 0
    for i in range(N):
        for j in range(N):
            if distances[i][j] > max_dist:
                max_dist = distances[i][j]
    
    print(f"\nNetwork diameter: {max_dist}")
    
    # 六度分隔分析
    six_degree_stats = stats.get(6, {})
    print(f"\nSix Degrees of Separation Analysis:")
    print(f"- {six_degree_stats['universal_ratio']:.1%} of people can reach everyone within 6 degrees")
    print(f"- {six_degree_stats['pair_ratio']:.1%} of all pairs are connected within 6 degrees")

if __name__ == "__main__":
    simple_demo()
