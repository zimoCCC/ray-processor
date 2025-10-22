# -*- coding: utf-8 -*-
"""
Ray Model Worker
简化的模型工作器，只需要继承基础模型类即可
"""
import ray
import logging
from typing import List, Dict, Any, Union
from models import BaseModel

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0.1)
class ModelWorker:
    """
    Ray Actor，每个GPU一个worker
    简化的实现，只需要指定模型类即可
    """
    
    def __init__(self, model_class, model_name: str, model_path: str = None, **kwargs):
        """
        初始化模型工作器
        
        Args:
            model_class: 模型类（如BEATsModel）
            model_name: 模型名称
            model_path: 模型路径
            **kwargs: 其他参数
        """
        self.model_class = model_class
        self.model_name = model_name
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 创建模型实例，让模型类自己处理设备分配
            self.model = self.model_class(
                model_name=self.model_name,
                model_path=self.model_path,
                **self.kwargs
            )
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_batch(self, batch_data: List[Union[Dict[str, Any], str]]) -> List[Dict[str, Any]]:
        """
        处理一批数据
        
        Args:
            batch_data: 批次数据
            
        Returns:
            处理结果列表
        """
        try:
            # 使用模型的generate_batch方法
            results = self.model.generate_batch(batch_data)
            
            # 添加worker信息到metadata
            for result in results:
                if "metadata" in result:
                    result["metadata"]["worker_processed"] = True
                else:
                    result["metadata"] = {"worker_processed": True}
            
            return results
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model:
            info = self.model.get_model_info()
            return info
        return {"model_loaded": False}
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            self.model.cleanup()
            self.model = None


# 便捷函数：创建特定类型的模型工作器
def create_beats_worker(model_name: str, model_path: str = None, **kwargs):
    """创建BEATs模型工作器"""
    from models import BEATsModel
    return ModelWorker.remote(BEATsModel, model_name, model_path, **kwargs)


def create_model_worker(model_class, model_name: str, model_path: str = None, **kwargs):
    """创建通用模型工作器"""
    return ModelWorker.remote(model_class, model_name, model_path, **kwargs)
