# 六度分隔理论模拟程序

本程序通过生成不同类型的人际关系网络来验证六度分隔理论。

## 功能特性

### 网络生成模型
1. **随机网络 (Erdős–Rényi模型)**
   - 每个人随机与k个其他人建立连接
   - 特点：度分布均匀，低聚类系数，短平均路径长度

2. **小世界网络 (Watts-Strogatz模型)**
   - 规则网络基础上的随机重连
   - 特点：高聚类系数，短平均路径长度，最符合现实社交网络

3. **无标度网络 (Barabási-Albert模型)**
   - 优先连接机制：新节点倾向于连接高度数节点
   - 特点：幂律度分布，存在"超级节点"

### 关键指标
- **通用可达性 (Universal Reachability)**: 能够在k度内到达所有其他人的人数比例
- **节点对连通性 (Pair Connectivity)**: 距离≤k的节点对占总节点对的比例

### 算法优化
- 自适应算法选择：根据网络密度自动选择最优的最短路径算法
- 稀疏网络：多源BFS (O(V×E))
- 密集网络：Floyd-Warshall (O(V³))

## 快速开始

### 1. 环境准备
```bash
# 运行安装脚本
python setup.py
```

### 2. 快速演示
```bash
# 运行简单演示版本 (不需要matplotlib)
python run/simple_demo.py
```

### 3. 完整模拟
```bash
# 运行完整的模拟程序 (需要matplotlib)
python run/simulation.py
```

## 项目结构

```
Six_Degrees_of_Seperation/
├── run/
│   ├── simulation.py          # 主模拟程序
│   └── simple_demo.py         # 简单演示版本
├── utils/
│   ├── __init__.py
│   ├── network_generator.py   # 网络生成器类
│   ├── graph_algorithms.py    # 图算法模块
│   └── network_models_explanation.md  # 网络模型详细说明
├── output/                    # 输出目录（自动创建）
├── setup.py                   # 安装脚本
├── TODO.md
└── README.md
```

## 参数说明

- **N**: 网络中的人数（节点数）
- **k**: 每个人的平均连接数
- **network_type**: 网络类型
  - 'random': 随机网络
  - 'small_world': 小世界网络
  - 'scale_free': 无标度网络

## 输出说明

### 控制台输出
- 网络基本统计信息
- 连通性分析
- 可达性分析结果

### 图表输出
- 覆盖率分析图：展示不同度数下的通用可达性和节点对连通性
- 度分布图：展示网络的度分布特征

### 文件输出
- 详细的统计结果文本文件
- 高质量的图表文件（PNG格式）

## 理论背景

六度分隔理论最初由匈牙利作家Frigyes Karinthy在1929年提出，认为世界上任意两个人之间的关系链不会超过6个人。该理论后来通过多种实验得到验证，成为复杂网络理论的重要基础。

### 网络模型特点对比

| 特征 | 随机网络 | 小世界网络 | 无标度网络 |
|------|----------|------------|------------|
| 度分布 | 泊松分布 | 接近泊松分布 | 幂律分布 |
| 聚类系数 | 低 | 高 | 中等 |
| 平均路径长度 | 短 | 短 | 很短 |
| 鲁棒性 | 中等 | 高 | 对随机攻击鲁棒，对针对性攻击脆弱 |

## 算法复杂度

- **网络生成**: O(N×k)
- **最短路径计算**: 
  - 稀疏网络: O(N×E) 使用多源BFS
  - 密集网络: O(N³) 使用Floyd-Warshall
- **可达性分析**: O(N²)

## 使用示例

```python
# 创建模拟器
sim = SixDegreesSimulation(n=500, k=8, network_type='small_world', seed=42)

# 查看网络统计
sim.print_network_summary()

# 生成分析图表
sim.plot_coverage_analysis(max_degree=8)
sim.plot_degree_distribution()

# 导出结果
sim.export_results('results.txt')
```

## 常见问题

**Q: 为什么小世界网络最符合现实？**
A: 小世界网络同时具有高聚类系数（体现局部聚集特性）和短平均路径长度（体现全局连通性），这与现实世界的社交网络特征最为相符。

**Q: 程序运行很慢怎么办？**
A: 可以调整参数N（人数）和k（连接数），或者使用simple_demo.py进行快速测试。

**Q: 如何理解通用可达性？**
A: 通用可达性指的是能够在k度内到达网络中所有其他节点的节点比例。这反映了网络中"超级连接者"的存在。

## 贡献

欢迎提交bug报告和功能建议！

## 许可证

MIT License
