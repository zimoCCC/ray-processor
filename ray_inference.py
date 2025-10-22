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
from tqdm import tqdm
import sqlite3

from dataloader import AudioJSONLDataLoader
from beats_model import BEATsModel

# 设置日志
def setup_logging(log_file=None):
    """设置日志配置"""
    if log_file:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    else:
        logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)




@ray.remote(num_gpus=0.1)
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
            # 保留原始输入的全部信息，仅在其基础上新增字段
            if isinstance(input_item, dict):
                result = dict(input_item)
            else:
                result = {"input": input_item}
            result["output"] = output
            result["metadata"] = {
                "worker_id": self.device_id,
                "batch_size": len(batch_data)
            }
            results.append(result)
        
        return results


def save_batch(batch_results, output_path, db_path=None, processed_lines=None):
    """保存一批结果到JSONL文件和SQLite数据库"""
    # 保存到JSONL文件
    with open(output_path, 'a', encoding='utf-8') as f:
        for result in batch_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 保存进度到SQLite数据库
    if db_path and processed_lines is not None:
        save_progress(db_path, processed_lines)


def save_progress(db_path, processed_lines):
    """保存处理进度到SQLite数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表（如果不存在）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY,
            processed_lines INTEGER
        )
    ''')
    
    # 更新进度
    cursor.execute('SELECT COUNT(*) FROM progress')
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO progress (processed_lines) VALUES (?)', (processed_lines,))
    else:
        cursor.execute('UPDATE progress SET processed_lines = ? WHERE id = 1', (processed_lines,))
    
    conn.commit()
    conn.close()


def get_progress(db_path):
    """获取处理进度"""
    if not os.path.exists(db_path):
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT processed_lines FROM progress WHERE id = 1')
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else 0


def run_inference(data_path, output_path, num_gpus=2, batch_size=4, model_path=None, db_path=None, enable_sqlite=True):
    """运行分布式推理"""
    
    # 计算worker数量
    gpu_per_worker = 0.1  # 每个worker占用的GPU数
    num_workers = int(num_gpus / gpu_per_worker)
    
    # 初始化Ray
    ray.init(num_gpus=num_gpus)
    
    try:
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 设置输出文件路径
        jsonl_path = os.path.join(output_path, "results.jsonl")
        db_path = os.path.join(output_path, "progress.db")
        log_path = os.path.join(output_path, "inference.log")
        
        # 设置日志文件
        setup_logging(log_path)
        
        # 检查断点续跑
        start_line = 0
        if enable_sqlite and db_path:
            start_line = get_progress(db_path)
            if start_line > 0:
                logger.info(f"Resuming from line {start_line}")
        
        # 创建数据加载器
        data_loader = AudioJSONLDataLoader(data_path, batch_size=batch_size)
        logger.info(f"Loaded {len(data_loader)} samples")
        
        # 创建worker
        workers = []
        for i in range(num_workers):
            worker = ModelWorker.remote("beats", i, model_path)
            workers.append(worker)
        
        logger.info(f"Total GPUs: {num_gpus}, Workers: {num_workers}, GPU per worker: {gpu_per_worker}")
        
        # 清空输出文件
        with open(jsonl_path, 'w') as f:
            pass
        
        # 分发数据并处理
        batch_count = 0
        total_saved = 0
        processed_lines = start_line
        
        # 创建进度条
        total_batches = len(data_loader)
        pbar = tqdm(total=total_batches, desc="Processing batches", unit="batch")
        
        # 分批并行处理
        chunk_size = num_workers * 8  # 每批处理8倍worker数量的任务
        data_iter = iter(data_loader)
        
        while True:
            # 提交一批任务
            futures = []
            chunk_batches = []
            
            for _ in range(chunk_size):
                try:
                    batch = next(data_iter)
                    worker_idx = batch_count % num_workers
                    worker = workers[worker_idx]
                    future = worker.generate_batch.remote(batch)
                    futures.append(future)
                    chunk_batches.append(batch)
                    batch_count += 1
                except StopIteration:
                    break
            
            if not futures:  # 没有更多数据
                break
            
            # 收集这批任务的结果
            for i, future in enumerate(futures):
                batch_results = ray.get(future)
                processed_lines += len(batch_results)
                save_batch(batch_results, jsonl_path, db_path if enable_sqlite else None, processed_lines)
                total_saved += len(batch_results)
                
                # 更新进度条
                pbar.update(len(batch_results))
                pbar.set_postfix({
                    'saved': total_saved,
                    'chunk': len(futures),
                    'worker': (batch_count - len(futures) + i) % num_workers
                })
        
        pbar.close()
        
        # 统计信息
        logger.info(f"Completed: {total_saved} samples, {batch_count} batches, {num_workers} workers")
        logger.info(f"Results saved to: {jsonl_path}")
        if enable_sqlite:
            logger.info(f"Progress saved to: {db_path}")
        logger.info(f"Logs saved to: {log_path}")
        
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
    output_path = "./outputs"
    enable_sqlite = True
    
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        model_path = sys.argv[model_idx + 1]
    else:
        print("Error: --model path is required")
        sys.exit(1)
    
    if "--output" in sys.argv:
        output_idx = sys.argv.index("--output")
        output_path = sys.argv[output_idx + 1]
    
    if "--no-sqlite" in sys.argv:
        enable_sqlite = False
    
    # 创建示例数据
    data_path = "/inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/data/shard_00.jsonl"
    
    if not os.path.exists(data_path):
        create_sample_data(data_path, 20)
    
    # 运行推理
    start_time = time.time()
    run_inference(
        data_path=data_path,
        output_path=output_path,
        num_gpus=8,
        batch_size=4,
        model_path=model_path,
        enable_sqlite=enable_sqlite
    )
