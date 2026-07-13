import time
import os
import sys

# ========== 1. 声明当前脚本所属的包 ==========
# 作用：让 Python 知道这个脚本是 "trainer" 包的一部分
# 为什么需要：直接运行脚本时（如 python train_pretrain.py），
# 如果不声明，Python 可能无法正确识别相对导入（如 from ..xxx import yyy），
# 导致 ModuleNotFoundError
__package__ = "trainer"

# ========== 2. 将项目根目录添加到 Python 模块搜索路径 ==========
# 作用：让 Python 能导入项目根目录下的其他模块（如 model/、dataset/）
#   - __file__: 当前脚本的完整路径，如 /path/to/minimind/trainer/train_pretrain.py
#   - os.path.dirname(__file__): 获取脚本所在目录，如 /path/to/minimind/trainer/
#   - os.path.join(..., '..'): 拼接上一级目录，得到 /path/to/minimind/trainer/.. => /path/to/minimind/
#   - os.path.abspath(...): 规范化为绝对路径，得到 /path/to/minimind
#   - sys.path.append(...): 将项目根目录添加到 Python 搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
