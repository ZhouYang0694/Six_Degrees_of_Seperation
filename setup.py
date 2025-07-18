"""
依赖项安装脚本
"""

import subprocess
import sys
import os

def install_requirements():
    """安装必要的Python包"""
    
    packages = [
        'numpy',
        'matplotlib',
        'networkx'  # 可选，用于对比验证
    ]
    
    print("Installing required packages...")
    print("="*40)
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    print("\n✓ All packages installed successfully!")
    return True

def test_imports():
    """测试导入"""
    print("\nTesting imports...")
    print("-" * 20)
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import networkx as nx
        print("✓ networkx imported successfully")
    except ImportError as e:
        print(f"✗ networkx import failed: {e}")
        print("  (networkx is optional)")
    
    return True

def main():
    """主函数"""
    print("Six Degrees of Separation - Setup Script")
    print("="*50)
    
    if install_requirements():
        if test_imports():
            print("\n🎉 Setup completed successfully!")
            print("\nYou can now run:")
            print("1. python run/simple_demo.py (quick demo)")
            print("2. python run/simulation.py (full simulation)")
        else:
            print("\n❌ Setup failed during import testing")
    else:
        print("\n❌ Setup failed during package installation")

if __name__ == "__main__":
    main()
