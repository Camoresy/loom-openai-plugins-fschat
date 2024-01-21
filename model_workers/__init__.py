import os
import sys

# 为了能使用fastchat_wrapper.py中的函数，需要将当前目录加入到sys.path中
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
from .base import *
from .zhipu import ChatGLMWorker
from .minimax import MiniMaxWorker
from .xinghuo import XingHuoWorker
from .qianfan import QianFanWorker
from .fangzhou import FangZhouWorker
from .qwen import QwenWorker
from .baichuan import BaiChuanWorker
from .azure import AzureWorker
from .tiangong import TianGongWorker
