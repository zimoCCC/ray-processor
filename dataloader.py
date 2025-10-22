"""
DataLoader类定义，用于数据加载和批处理
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    基础数据加载器类
    提供通用的数据访问和断点续跑支持
    """

    def __init__(self, data_path: str, batch_size: int = 1, shuffle: bool = False, **kwargs):
        """
        初始化数据加载器

        Args:
            data_path: 数据文件路径
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            **kwargs: 其他参数
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.config = kwargs
        self.data = []  # 数据存储
        self._current_position = 0  # 当前位置

        logger.info(f"Initializing DataLoader with path: {data_path}")
        self._load_data()

    @abstractmethod
    def _load_data(self):
        """加载数据，子类需要实现"""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """迭代器，返回批次数据"""
        pass

    def __len__(self) -> int:
        """返回数据总数"""
        return len(self.data)

    def get_batch(self, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """
        获取指定范围的数据批次

        Args:
            start_idx: 开始索引
            end_idx: 结束索引

        Returns:
            数据批次
        """
        return self.data[start_idx:end_idx]

    def set_position(self, position: int):
        """
        设置数据加载器的位置（用于断点续跑）

        Args:
            position: 位置索引
        """
        if position < 0 or position >= len(self.data):
            raise IndexError(f"Position {position} out of range [0, {len(self.data)})")
        self._current_position = position

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        根据索引获取单个数据项

        Args:
            index: 数据索引

        Returns:
            数据项
        """
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range [0, {len(self.data)})")
        return self.data[index]

    def get_batch_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        根据索引列表获取数据批次

        Args:
            indices: 索引列表

        Returns:
            数据批次
        """
        batch = []
        for idx in indices:
            if 0 <= idx < len(self.data):
                batch.append(self.data[idx])
            else:
                logger.warning(f"Index {idx} out of range, skipping")
        return batch

    def get_current_position(self) -> int:
        """获取当前位置"""
        return self._current_position

    def reset_position(self):
        """重置位置到开始"""
        self._current_position = 0


class JSONLDataLoader(BaseDataLoader):
    """
    JSONL格式数据加载器
    """

    def _load_data(self):
        """加载JSONL数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.data = []
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    item = json.loads(line)
                    self.data.append(item)

        logger.info(f"Loaded {len(self.data)} items from {self.data_path}")


class AudioJSONLDataLoader(BaseDataLoader):
    """
    音频JSONL数据加载器，专门处理音频文件路径
    """

    def _load_data(self):
        """加载音频JSONL数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.data = []
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    item = json.loads(line)
                    # 确保有audio_path字段
                    if 'audio_path' not in item:
                        if 'prompt' in item:
                            item['audio_path'] = item['prompt']
                        if 'path' in item:
                            item['audio_path'] = item['path']
                        elif 'file' in item:
                            item['audio_path'] = item['file']
                        else:
                            logger.warning(f"No audio_path found in item: {item}")
                    self.data.append(item)

        logger.info(f"Loaded {len(self.data)} audio items from {self.data_path}")

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """迭代器实现"""
        if self.shuffle:
            import random

            indices = list(range(len(self.data)))
            random.shuffle(indices)
        else:
            indices = list(range(len(self.data)))

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.data[idx] for idx in batch_indices]
            yield batch

    # 继承基类的所有通用方法，无需重复实现


if __name__ == '__main__':
    # 测试JSONL数据加载器
    import os
    import tempfile

    print('Testing JSONL DataLoader...')

    # 创建临时JSONL文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name

        # 写入测试数据
        test_data = [
            {'id': '1', 'text': 'Hello world', 'label': 'greeting'},
            {'id': '2', 'text': 'How are you?', 'label': 'question'},
            {'id': '3', 'text': 'Good morning', 'label': 'greeting'},
            {'id': '4', 'text': "What's the weather?", 'label': 'question'},
            {'id': '5', 'text': 'Thank you', 'label': 'gratitude'},
        ]

        for item in test_data:
            f.write(json.dumps(item) + '\n')

    try:
        # 测试数据加载器
        loader = JSONLDataLoader(temp_path, batch_size=2)

        print(f"Total samples: {len(loader)}")
        print(f"Expected: 5")

        # 测试批次迭代
        for i, batch in enumerate(loader):
            print(f"Batch {i+1}: {len(batch)} samples")
            for j, item in enumerate(batch):
                print(f"  Sample {j+1}: {item['id']} - {item['text']}")

        print('✅ JSONL DataLoader test passed!')

    except Exception as e:
        print(f"❌ Test failed: {e}")

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
