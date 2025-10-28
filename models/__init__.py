# -*- coding: utf-8 -*-
"""
模型模块
提供统一的模型接口和实现
"""

from .base_model import BaseModel
from .beats_model import BEATsModel
from .qwen3omni_model import Qwen3OmniModel

__all__ = ['BaseModel', 'BEATsModel', 'Qwen3OmniModel']