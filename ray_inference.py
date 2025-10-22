# -*- coding: utf-8 -*-
"""
Ray分布式推理实现
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
from models import BEATsModel
from task_tracker import TaskTracker
from model_worker import create_beats_worker

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




# ModelWorker类已移至 model_worker.py 模块


def save_batch(batch_results, output_path):
    """保存一批结果到JSONL文件"""
    # 保存到JSONL文件
    with open(output_path, 'a', encoding='utf-8') as f:
        for result in batch_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


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
        
        # 创建数据加载器
        data_loader = AudioJSONLDataLoader(data_path, batch_size=batch_size)
        total_samples = len(data_loader)
        logger.info(f"Loaded {total_samples} samples")
        
        # 初始化任务跟踪器
        task_tracker = None
        if enable_sqlite:
            task_tracker = TaskTracker(db_path)
            task_tracker.init_tasks(total_samples)
            # 获取已完成的任务
            completed_tasks = task_tracker.get_completed_tasks()
            logger.info(f"Found {len(completed_tasks)} completed tasks, resuming from task {len(completed_tasks)}")
        else:
            completed_tasks = set()
        
        # 创建worker
        workers = []
        for i in range(num_workers):
            worker = create_beats_worker("beats", model_path)
            workers.append(worker)
        
        logger.info(f"Total GPUs: {num_gpus}, Workers: {num_workers}, GPU per worker: {gpu_per_worker}")
        
        # 清空输出文件（如果不是断点续跑）
        if not completed_tasks:
            with open(jsonl_path, 'w') as f:
                pass
        
        # 分发数据并处理
        batch_count = 0
        total_saved = 0
        
        # 创建进度条
        remaining_tasks = total_samples - len(completed_tasks)
        pbar = tqdm(total=remaining_tasks, desc="Processing samples", unit="sample")
        
        # 分批并行处理
        chunk_size = num_workers * 8  # 每批处理8倍worker数量的任务
        data_iter = iter(data_loader)
        
        while True:
            # 获取未分配的任务
            if enable_sqlite and task_tracker:
                unallocated_task_ids = task_tracker.get_unallocated_tasks(chunk_size)
                if not unallocated_task_ids:
                    logger.info("No more unallocated tasks")
                    break
                
                # 标记这些任务为已分配
                worker_id = f"worker_{batch_count % num_workers}"
                task_tracker.mark_tasks_allocated(unallocated_task_ids, worker_id)
                
                # 获取对应的数据批次
                batch_data = data_loader.get_batch_by_indices(unallocated_task_ids)
                
                if not batch_data:
                    break
            else:
                # 不使用SQLite时的简单处理
                try:
                    batch_data = next(data_iter)
                except StopIteration:
                    break
            
            # 提交任务到worker
            futures = []
            task_ids = unallocated_task_ids if enable_sqlite else list(range(batch_count, batch_count + len(batch_data)))
            
            for i, data_item in enumerate(batch_data):
                worker_idx = (batch_count + i) % num_workers
                worker = workers[worker_idx]
                future = worker.generate_batch.remote([data_item])  # 单个数据项
                futures.append((future, task_ids[i]))
            
            # 收集结果
            completed_task_ids = []
            for future, task_id in futures:
                try:
                    batch_results = ray.get(future)
                    if batch_results:
                        # 保存结果
                        save_batch(batch_results, jsonl_path)
                        total_saved += len(batch_results)
                        completed_task_ids.append(task_id)
                        
                        # 更新进度条
                        pbar.update(len(batch_results))
                        pbar.set_postfix({
                            'saved': total_saved,
                            'completed': len(completed_tasks) + total_saved if enable_sqlite else total_saved,
                            'total': total_samples
                        })
                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {e}")
                    continue
            
            # 批量标记任务为已完成
            if enable_sqlite and task_tracker and completed_task_ids:
                task_tracker.mark_tasks_completed(completed_task_ids)
            
            batch_count += 1
        
        pbar.close()
        
        # 显示最终统计信息
        if enable_sqlite and task_tracker:
            stats = task_tracker.get_progress_stats()
            logger.info(f"Final stats: {stats}")
        
        logger.info(f"Completed: {total_saved} samples, {batch_count} batches, {num_workers} workers")
        logger.info(f"Results saved to: {jsonl_path}")
        if enable_sqlite:
            logger.info(f"Progress saved to: {db_path}")
        logger.info(f"Logs saved to: {log_path}")
        
    finally:
        ray.shutdown()




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
    
    # 数据路径
    data_path = "/inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/data/shard_00.jsonl"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
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
