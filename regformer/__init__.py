# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:__init__.py
# @Software:PyCharm
# @Created Time:2024/4/8 4:05 PM
__version__ = "0.0.1"
from .model import BiMamba, mambaLM
import logging
import sys

logger = logging.getLogger("scMamba")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

