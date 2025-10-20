# -*- coding: utf-8 -*-
"""
简单的Ray分布式推理实现
"""
import ray
import json
import time
import logging
from typing import List, Dict, Any
import os
from pathlib import Path

from dataloader import AudioJSONLDataLoader
from beats_model import BEATsModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




@ray.remote(num_gpus=1)
class ModelWorker:
    """Ray Actor，每个GPU一个worker"""
    
    def __init__(self, model_name, device_id, model_path=None):
        self.model_name = model_name
        self.device_id = device_id
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """在Ray worker中加载模型"""
        self.model = BEATsModel(
            model_name=self.model_name,
            device=f"cuda:{self.device_id}" if self.device_id >= 0 else "cpu",
            model_path=self.model_path
        )
    
    def generate_batch(self, batch_data):
        """处理一批数据"""
        # 提取音频路径
        inputs = []
        for item in batch_data:
            if isinstance(item, dict):
                audio_path = item.get('audio_path', '') or item.get('path', '') or item.get('file', '')
                inputs.append(audio_path)
            else:
                inputs.append(str(item))
        
        # 生成结果
        outputs = self.model.generate(inputs)
        
        # 构建结果
        results = []
        for i, (input_item, output) in enumerate(zip(batch_data, outputs)):
            result = {
                "input": input_item,
                "output": output,
                "metadata": {
                    "worker_id": self.device_id,
                    "batch_size": len(batch_data)
                }
            }
            results.append(result)
        
        return results


def save_batch(batch_results, output_path):
    """保存一批结果到JSONL文件"""
    with open(output_path, 'a', encoding='utf-8') as f:
        for result in batch_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def run_inference(data_path, output_path, num_gpus=2, batch_size=4, model_path=None):
    """运行分布式推理"""
    
    # 初始化Ray
    ray.init(num_gpus=num_gpus)
    
    try:
        # 创建数据加载器
        data_loader = AudioJSONLDataLoader(data_path, batch_size=batch_size)
        logger.info(f"Loaded {len(data_loader)} samples")
        
        # 创建worker
        workers = []
        for i in range(num_gpus):
            worker = ModelWorker.remote("beats", i, model_path)
            workers.append(worker)
        
        # 清空输出文件
        with open(output_path, 'w') as f:
            pass
        
        # 分发数据并处理
        batch_count = 0
        total_saved = 0
        
        for batch in data_loader:
            # 轮询分配任务给worker
            worker_idx = batch_count % num_gpus
            worker = workers[worker_idx]
            
            # 提交任务并立即收集结果
            future = worker.generate_batch.remote(batch)
            batch_results = ray.get(future)
            save_batch(batch_results, output_path)
            
            batch_count += 1
            total_saved += len(batch_results)
            logger.info(f"Processed batch {batch_count}, total saved: {total_saved}")
        
        # 统计信息
        logger.info(f"Completed: {total_saved} samples, {batch_count} batches, {num_gpus} workers")
        
    finally:
        ray.shutdown()


def create_sample_data(data_path, num_samples=20):
    """创建示例音频数据"""
    sample_data = []
    for i in range(num_samples):
        sample = {
            "id": f"audio_{i}",
            "audio_path": f"/path/to/audio_{i}.wav",
            "metadata": {
                "duration": 10.0 + i * 0.5,
                "sample_rate": 16000
            }
        }
        sample_data.append(sample)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Created {num_samples} audio samples at {data_path}")


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    model_path = None
    if "--model" in sys.argv:
        try:
            model_idx = sys.argv.index("--model")
            model_path = sys.argv[model_idx + 1]
        except (ValueError, IndexError):
            print("Error: --model requires a path")
            sys.exit(1)
    else:
        print("Error: --model path is required")
        sys.exit(1)
    
    # 创建示例数据
    data_path = "sample_data.jsonl"
    output_path = "results.jsonl"
    
    if not os.path.exists(data_path):
        create_sample_data(data_path, 20)
    
    # 运行推理
    start_time = time.time()
    run_inference(
        data_path=data_path,
        output_path=output_path,
        num_gpus=2,  # 使用2个GPU
        batch_size=4,
        model_path=model_path
    )
