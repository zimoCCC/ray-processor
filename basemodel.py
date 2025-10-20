"""
BaseModel类定义，提供模型的基础接口
"""
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    基础模型类，定义模型的标准接口
    """
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """
        初始化模型
        
        Args:
            model_name: 模型名称
            device: 设备类型 (cuda/cpu)
            **kwargs: 其他模型参数
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = kwargs
        
        logger.info(f"Initializing {model_name} on {device}")
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """
        加载模型和tokenizer，子类需要实现
        """
        pass
    
    @abstractmethod
    def generate(self, 
                 inputs: Union[str, List[str]], 
                 max_length: int = 512,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 **kwargs) -> Union[str, List[str]]:
        """
        生成文本
        
        Args:
            inputs: 输入文本或文本列表
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: nucleus sampling参数
            do_sample: 是否采样
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本或文本列表
        """
        pass
    
    def preprocess(self, inputs: Union[str, List[str]]) -> Any:
        """
        预处理输入数据
        
        Args:
            inputs: 输入文本或文本列表
            
        Returns:
            预处理后的数据
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # 默认实现：使用tokenizer编码
        if self.tokenizer is not None:
            return self.tokenizer(inputs, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True)
        return inputs
    
    def to_device(self, data: Any) -> Any:
        """
        将数据移动到指定设备
        
        Args:
            data: 要移动的数据
            
        Returns:
            移动后的数据
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)([self.to_device(item) for item in data])
        return data
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "config": self.config,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }