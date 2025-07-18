"""
Six Degrees of Separation Simulation
六度分隔理论验证程序

该程序模拟人际关系网络并验证六度分隔理论
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

from utils.network_generator import NetworkGenerator
from utils.graph_algorithms import (
    optimized_all_pairs_shortest_path,
    calculate_reachability_stats,
    analyze_network_connectivity,
    get_diameter,
    get_average_path_length
)


class SixDegreesSimulation:
    """
    六度分隔理论模拟器
    """
    
    def __init__(self, n: int, k: int, network_type: str = 'random', seed: int = None):
        """
        初始化模拟器
        
        Args:
            n: 人数
            k: 每个人的平均连接数
            network_type: 网络类型 ('random', 'small_world', 'scale_free')
            seed: 随机种子
        """
        self.n = n
        self.k = k
        self.network_type = network_type
        self.seed = seed
        
        # 生成网络
        self.generator = NetworkGenerator(n, k, seed)
        print(f"Generating {network_type} network with {n} nodes and average degree {k}...")
        
        start_time = time.time()
        
        if network_type == 'random':
            self.edges = self.generator.generate_random_network()
        elif network_type == 'small_world':
            self.edges = self.generator.generate_small_world_network()
        elif network_type == 'scale_free':
            self.edges = self.generator.generate_scale_free_network()
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        generation_time = time.time() - start_time
        print(f"Network generated in {generation_time:.2f} seconds")
        
        # 网络统计
        self.network_stats = self.generator.get_network_stats(self.edges)
        self.connectivity_stats = analyze_network_connectivity(self.edges, n)
        
        # 计算最短路径
        print("Calculating shortest paths...")
        start_time = time.time()
        self.distance_matrix = optimized_all_pairs_shortest_path(self.edges, n)
        calculation_time = time.time() - start_time
        print(f"Shortest paths calculated in {calculation_time:.2f} seconds")
        
        # 计算可达性统计
        print("Calculating reachability statistics...")
        self.reachability_stats = calculate_reachability_stats(self.distance_matrix, max_degree=10)
        
        # 网络特征
        self.diameter = get_diameter(self.distance_matrix)
        self.avg_path_length = get_average_path_length(self.distance_matrix)
    
    def print_network_summary(self):
        """
        打印网络摘要信息
        """
        print("\n" + "="*50)
        print("NETWORK SUMMARY")
        print("="*50)
        print(f"Network Type: {self.network_type}")
        print(f"Number of nodes (N): {self.n}")
        print(f"Target average degree (k): {self.k}")
        print(f"Actual average degree: {self.network_stats['average_degree']:.2f}")
        print(f"Total edges: {self.network_stats['total_edges']}")
        print(f"Network diameter: {self.diameter}")
        print(f"Average path length: {self.avg_path_length:.2f}")
        print(f"Largest component size: {self.connectivity_stats['largest_component_size']}")
        print(f"Connectivity ratio: {self.connectivity_stats['connectivity_ratio']:.3f}")
        print(f"Number of components: {self.connectivity_stats['num_components']}")
        
        print("\n" + "-"*30)
        print("REACHABILITY ANALYSIS")
        print("-"*30)
        
        for degree in range(1, 7):
            stats = self.reachability_stats.get(degree, {})
            if stats:
                print(f"Within {degree} degree(s):")
                print(f"  Universal reachability: {stats['universal_coverage']:.1%} "
                      f"({stats['universal_reachable_nodes']}/{stats['total_nodes']} people can reach everyone)")
                print(f"  Pair connectivity: {stats['pair_coverage']:.1%} "
                      f"({stats['reachable_pairs']}/{stats['total_pairs']} pairs connected)")
    
    def plot_coverage_analysis(self, max_degree: int = 6, save_path: str = None):
        """
        绘制覆盖率分析图
        
        Args:
            max_degree: 最大度数
            save_path: 保存路径
        """
        degrees = list(range(1, max_degree + 1))
        universal_coverage = []
        pair_coverage = []
        
        for degree in degrees:
            stats = self.reachability_stats.get(degree, {})
            universal_coverage.append(stats.get('universal_coverage', 0))
            pair_coverage.append(stats.get('pair_coverage', 0))
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(degrees, universal_coverage, 'b-o', linewidth=2, markersize=8, 
                label='Universal Reachability Ratio\n(People who can reach everyone within k degrees)')
        plt.plot(degrees, pair_coverage, 'r-s', linewidth=2, markersize=8,
                label='Pair Connectivity Ratio\n(Pairs connected within k degrees)')
        
        plt.xlabel('Degrees of Separation (k)', fontsize=12)
        plt.ylabel('Coverage Ratio', fontsize=12)
        plt.title(f'Six Degrees of Separation Analysis\n'
                 f'Network: {self.network_type.title()}, N={self.n}, k={self.k}',
                 fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(0.5, max_degree + 0.5)
        plt.ylim(0, 1.05)
        
        # 添加六度分隔线
        plt.axvline(x=6, color='g', linestyle='--', alpha=0.7, linewidth=2,
                   label='Six Degrees')
        plt.legend(fontsize=11)
        
        # 添加数值标签
        for i, (degree, univ_cov, pair_cov) in enumerate(zip(degrees, universal_coverage, pair_coverage)):
            plt.annotate(f'{univ_cov:.1%}', (degree, univ_cov), 
                        textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(f'{pair_cov:.1%}', (degree, pair_cov), 
                        textcoords="offset points", xytext=(0,-15), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_degree_distribution(self, save_path: str = None):
        """
        绘制度分布图
        
        Args:
            save_path: 保存路径
        """
        degrees = self.network_stats['degree_distribution']
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), 
                alpha=0.7, edgecolor='black')
        plt.xlabel('Degree', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Degree Distribution\n'
                 f'Network: {self.network_type.title()}, N={self.n}, k={self.k}',
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加平均度的垂直线
        plt.axvline(x=self.network_stats['average_degree'], 
                   color='r', linestyle='--', linewidth=2,
                   label=f'Average Degree: {self.network_stats["average_degree"]:.2f}')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Degree distribution plot saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str):
        """
        导出结果到文件
        
        Args:
            filename: 文件名
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Six Degrees of Separation Simulation Results\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Network Type: {self.network_type}\n")
            f.write(f"Number of nodes (N): {self.n}\n")
            f.write(f"Target average degree (k): {self.k}\n")
            f.write(f"Actual average degree: {self.network_stats['average_degree']:.2f}\n")
            f.write(f"Total edges: {self.network_stats['total_edges']}\n")
            f.write(f"Network diameter: {self.diameter}\n")
            f.write(f"Average path length: {self.avg_path_length:.2f}\n")
            f.write(f"Connectivity ratio: {self.connectivity_stats['connectivity_ratio']:.3f}\n\n")
            
            f.write("Reachability Analysis:\n")
            f.write("-" * 30 + "\n")
            
            for degree in range(1, 7):
                stats = self.reachability_stats.get(degree, {})
                if stats:
                    f.write(f"Within {degree} degree(s):\n")
                    f.write(f"  Universal reachability: {stats['universal_coverage']:.1%} "
                           f"({stats['universal_reachable_nodes']}/{stats['total_nodes']} people can reach everyone)\n")
                    f.write(f"  Pair connectivity: {stats['pair_coverage']:.1%} "
                           f"({stats['reachable_pairs']}/{stats['total_pairs']} pairs connected)\n")
        
        print(f"Results exported to: {filename}")


def main():
    """
    主程序
    """
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 参数设置
    N = 5000   # 人数 (降低以提高演示速度)
    k = 8     # 平均连接数
    
    print("Six Degrees of Separation Simulation")
    print("="*50)
    print(f"Parameters: N={N}, k={k}")
    print("\nNetwork Generation Methods:")
    print("1. Random Network (Erdős–Rényi): Each person randomly connects to k others")
    print("2. Small World Network (Watts-Strogatz): Regular network with random rewiring")
    print("3. Scale-Free Network (Barabási-Albert): Preferential attachment growth")
    print("\nMetrics:")
    print("- Universal Reachability: People who can reach everyone within k degrees")
    print("- Pair Connectivity: Fraction of all possible pairs connected within k degrees")
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同类型的网络
    network_types = ['random', 'small_world', 'scale_free']
    
    for network_type in network_types:
        print(f"\n{'='*60}")
        print(f"Simulating {network_type.upper()} network...")
        print(f"{'='*60}")
        
        try:
            # 创建模拟器
            sim = SixDegreesSimulation(N, k, network_type, seed=42)
            
            # 打印网络摘要
            sim.print_network_summary()
            
            # 绘制覆盖率分析图
            coverage_path = os.path.join(output_dir, f'{network_type}_coverage_analysis.png')
            sim.plot_coverage_analysis(max_degree=8, save_path=coverage_path)
            
            # 绘制度分布图
            degree_path = os.path.join(output_dir, f'{network_type}_degree_distribution.png')
            sim.plot_degree_distribution(save_path=degree_path)
            
            # 导出结果
            results_path = os.path.join(output_dir, f'{network_type}_results.txt')
            sim.export_results(results_path)
            
        except Exception as e:
            print(f"Error simulating {network_type} network: {e}")
            continue
    
    print("\nSimulation completed! Check the output folder for results.")
    print("\nKey Insights:")
    print("- Random networks: Fast global connectivity but low local clustering")
    print("- Small-world networks: Best balance of local clustering and global connectivity")
    print("- Scale-free networks: Highly connected hubs enable fast information spread")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()