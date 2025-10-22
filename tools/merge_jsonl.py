#!/usr/bin/env python3
"""
JSONL文件合并工具
递归查找目录下的所有JSONL文件，提取prompt和output字段并合并
支持多进程处理和进度条显示
"""

import argparse
import json
import logging
import os
from multiprocessing import Manager, Pool, Process, cpu_count
from pathlib import Path

from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_QUEUE = None  # 将在工作进程初始化时注入


def _init_worker(queue_ref):
    """在工作进程中设置全局队列引用。"""
    global _QUEUE
    _QUEUE = queue_ref


def _writer_process(queue_ref, output_file, sentinel):
    """
    专用写入进程：从队列读取条目并写入JSONL文件，直到接收到sentinel。
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        while True:
            item = queue_ref.get()
            if item == sentinel:
                break
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def find_jsonl_files(directory):
    """
    递归查找目录下的所有JSONL文件

    Args:
        directory (str): 要搜索的目录路径

    Returns:
        list: JSONL文件路径列表
    """
    jsonl_files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.error(f"目录不存在: {directory}")
        return jsonl_files

    # 递归查找所有.jsonl文件
    for file_path in directory_path.rglob('*.jsonl'):
        if file_path.is_file():
            jsonl_files.append(str(file_path))

    logger.info(f"找到 {len(jsonl_files)} 个JSONL文件")
    return jsonl_files


def process_jsonl_file(file_path):
    """
    处理单个JSONL文件，提取prompt和output字段

    Args:
        file_path (str): JSONL文件路径

    Returns:
        tuple: (处理的条目数, 有效条目数)
    """
    global _QUEUE
    processed_count = 0
    valid_count = 0

    try:
        with open(file_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                processed_count += 1

                try:
                    data = json.loads(line)

                    # 提取prompt和output字段
                    extracted_item = {}
                    if 'prompt' in data:
                        extracted_item['prompt'] = data['prompt']
                    if 'output' in data:
                        extracted_item['output'] = data['output']
                    if 'index' in data:
                        extracted_item['index'] = data['index']

                    # 只有当至少有一个字段存在时才添加
                    if extracted_item:
                        if _QUEUE is not None:
                            _QUEUE.put(extracted_item)
                        valid_count += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"文件 {file_path} 第 {line_num} 行JSON解析错误: {e}")
                    continue

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        return 0, 0

    return processed_count, valid_count


def merge_jsonl_files(input_directory, output_file, num_processes=None):
    """
    合并JSONL文件的主函数

    Args:
        input_directory (str): 输入目录
        output_file (str): 输出文件路径
        num_processes (int): 进程数，默认为CPU核心数
    """
    if num_processes is None:
        num_processes = cpu_count()

    logger.info(f"开始处理，使用 {num_processes} 个进程")

    # 查找所有JSONL文件
    jsonl_files = find_jsonl_files(input_directory)

    if not jsonl_files:
        logger.warning('没有找到任何JSONL文件')
        return

    # 启动写入进程并进行多进程解析
    logger.info('开始多进程处理文件与异步写入...')
    manager = Manager()
    queue_ref = manager.Queue(maxsize=10000)
    sentinel = ('__SENTINEL__',)
    writer = Process(target=_writer_process, args=(queue_ref, output_file, sentinel))
    writer.start()

    total_processed = 0
    total_valid = 0
    with Pool(processes=num_processes, initializer=_init_worker, initargs=(queue_ref,)) as pool:
        with tqdm(desc='处理条目', unit='条') as pbar:
            for processed_count, valid_count in pool.imap(process_jsonl_file, jsonl_files):
                total_processed += processed_count
                total_valid += valid_count
                if processed_count:
                    pbar.update(processed_count)
                pbar.set_postfix({'文件': len(jsonl_files), '已提取': total_valid})

    # 通知写入进程完成并等待退出
    queue_ref.put(sentinel)
    writer.join()

    logger.info(
        f"合并完成！总共处理了 {len(jsonl_files)} 个文件，{total_processed} 条原始记录，提取写入 {total_valid} 条有效记录"
    )
    logger.info(f"输出文件: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='递归查找并合并JSONL文件')
    parser.add_argument('input_directory', help='输入目录路径')
    parser.add_argument('output_file', help='输出文件路径')
    parser.add_argument('--processes', '-p', type=int, default=None, help='进程数，默认为CPU核心数')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细日志')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查输入目录
    if not os.path.exists(args.input_directory):
        logger.error(f"输入目录不存在: {args.input_directory}")
        return

    # 开始处理
    merge_jsonl_files(args.input_directory, args.output_file, args.processes)


if __name__ == '__main__':
    main()
