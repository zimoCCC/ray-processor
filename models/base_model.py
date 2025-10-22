"""
基础模型类
定义统一的模型接口
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    基础模型类，定义统一的模型接口

    所有模型都应该继承此类并实现必要的方法
    """

    def __init__(self, model_name: str, model_path: str = None, **kwargs):
        """
        初始化模型

        Args:
            model_name: 模型名称
            model_path: 模型路径
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.model_path = model_path
        self.config = kwargs
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        加载模型，子类必须实现
        """
        pass

    @abstractmethod
    def generate(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        生成结果，子类必须实现

        Args:
            inputs: 输入数据列表

        Returns:
            生成结果列表
        """
        pass

    def generate_batch(self, batch_data: List[Union[Dict[str, Any], str]]) -> List[Dict[str, Any]]:
        """
        处理一批数据，提供默认实现

        Args:
            batch_data: 批次数据，可以是字典列表或字符串列表

        Returns:
            处理结果列表
        """
        # 提取输入数据
        inputs = self._extract_inputs(batch_data)

        # 生成结果
        outputs = self.generate(inputs)

        # 构建结果
        results = []
        for i, (input_item, output) in enumerate(zip(batch_data, outputs)):
            # 保留原始输入的全部信息，仅在其基础上新增字段
            if isinstance(input_item, dict):
                result = dict(input_item)
            else:
                result = {'input': input_item}

            result['output'] = output
            result['metadata'] = {'model_name': self.model_name, 'batch_size': len(batch_data)}
            results.append(result)

        return results

    def _extract_inputs(self, batch_data: List[Union[Dict[str, Any], str]]) -> List[str]:
        """
        从批次数据中提取输入数据

        Args:
            batch_data: 批次数据

        Returns:
            输入数据列表
        """
        inputs = []
        for item in batch_data:
            if isinstance(item, dict):
                # 尝试从不同字段获取输入路径
                audio_path = (
                    item.get('audio_path', '')
                    or item.get('path', '')
                    or item.get('file', '')
                    or item.get('prompt', '')
                )
                inputs.append(audio_path)
            else:
                inputs.append(str(item))
        return inputs

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {'model_name': self.model_name, 'model_path': self.model_path, 'config': self.config}

    def cleanup(self):
        """
        清理模型资源，子类可以重写
        """
        pass

    def __del__(self):
        """析构函数，自动清理资源"""
        self.cleanup()
