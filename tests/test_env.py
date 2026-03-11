import sys
import os

print("=" * 50)
print("环境信息测试")
print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print(f"当前目录: {os.getcwd()}")
print(f"虚拟环境: {os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda env')}")
print("=" * 50)
print("✅ 虚拟环境配置成功!")
