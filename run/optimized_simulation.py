"""
优化版本的六度分隔理论模拟程序
解决了递归深度和性能问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import traceback
from typing import Dict, List, Tuple

# 尝试导入numpy和matplotlib，如果失败则使用替代方案
try:
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: numpy or matplotlib not available. Plotting disabled.")

from utils.network_generator import NetworkGenerator
from utils.graph_algorithms import (
    optimized_all_pairs_shortest_path,
    calculate_reachability_stats,
    analyze_network_connectivity,
    get_diameter,
    get_average_path_length
)


class OptimizedSixDegreesSimulation:
    """
    优化的六度分隔理论模拟器
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
        
        # 参数验证
        if n <= 0:
            raise ValueError("Number of nodes must be positive")
        if k <= 0:
            raise ValueError("Average degree must be positive")
        if k >= n:
            raise ValueError("Average degree must be less than number of nodes")
        
        # 生成网络
        self.generator = NetworkGenerator(n, k, seed)
        print(f"Generating {network_type} network with {n} nodes and average degree {k}...")
        
        start_time = time.time()
        
        try:
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
            print(f"Generated {len(self.edges)} edges")
            
        except Exception as e:
            print(f"Error generating network: {e}")
            raise
        
        # 网络统计
        try:
            self.network_stats = self.generator.get_network_stats(self.edges)
            self.connectivity_stats = analyze_network_connectivity(self.edges, n)
        except Exception as e:
            print(f"Error calculating network statistics: {e}")
            raise
        
        # 计算最短路径
        print("Calculating shortest paths...")
        start_time = time.time()
        
        try:
            self.distance_matrix = optimized_all_pairs_shortest_path(self.edges, n)
            calculation_time = time.time() - start_time
            print(f"Shortest paths calculated in {calculation_time:.2f} seconds")
        except Exception as e:
            print(f"Error calculating shortest paths: {e}")
            raise
        
        # 计算可达性统计
        print("Calculating reachability statistics...")
        try:
            self.reachability_stats = calculate_reachability_stats(self.distance_matrix, max_degree=10)
        except Exception as e:
            print(f"Error calculating reachability statistics: {e}")
            raise
        
        # 网络特征
        try:
            self.diameter = get_diameter(self.distance_matrix)
            self.avg_path_length = get_average_path_length(self.distance_matrix)
        except Exception as e:
            print(f"Error calculating network features: {e}")
            self.diameter = -1
            self.avg_path_length = -1
    
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
        绘制覆盖率分析图（如果matplotlib可用）
        """
        if not HAS_PLOTTING:
            print("Plotting not available. Install matplotlib to enable plotting.")
            return
        
        try:
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
            
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def export_results(self, filename: str):
        """
        导出结果到文件
        """
        try:
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
            
        except Exception as e:
            print(f"Error exporting results: {e}")


def main():
    """
    主程序
    """
    # 设置matplotlib支持中文（如果可用）
    if HAS_PLOTTING:
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("Warning: Could not set Chinese font for matplotlib")
    
    # 参数设置 - 使用保守的参数避免性能问题
    N = 5000   # 人数 (适中的规模)
    k = 8     # 平均连接数
    
    print("Optimized Six Degrees of Separation Simulation")
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
            sim = OptimizedSixDegreesSimulation(N, k, network_type, seed=42)
            
            # 打印网络摘要
            sim.print_network_summary()
            
            # 绘制覆盖率分析图
            coverage_path = os.path.join(output_dir, f'{network_type}_coverage_analysis.png')
            sim.plot_coverage_analysis(max_degree=8, save_path=coverage_path)
            
            # 导出结果
            results_path = os.path.join(output_dir, f'{network_type}_results.txt')
            sim.export_results(results_path)
            
        except Exception as e:
            print(f"Error simulating {network_type} network: {e}")
            traceback.print_exc()
            continue
    
    print("\nSimulation completed! Check the output folder for results.")
    print("\nKey Insights:")
    print("- Random networks: Fast global connectivity but low local clustering")
    print("- Small-world networks: Best balance of local clustering and global connectivity")
    print("- Scale-free networks: Highly connected hubs enable fast information spread")
    print("\nOptimizations applied:")
    print("- Iterative DFS instead of recursive to avoid stack overflow")
    print("- Efficient neighbor tracking in small-world network generation")
    print("- Adaptive algorithm selection based on network density")
    print("- Comprehensive error handling and performance monitoring")


if __name__ == "__main__":
    main()
