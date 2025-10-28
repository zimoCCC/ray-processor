"""
DataLoader类定义，用于数据加载和批处理
"""

import json
import logging
import os
import subprocess
from datetime import timedelta
from typing import Tuple
from contextlib import nullcontext
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


def _sanitize_endpoint(ep: str) -> str:
    ep = ep.strip()
    if ep.startswith("http://"):
        ep = ep[len("http://"):]
    elif ep.startswith("https://"):
        ep = ep[len("https://"):]
    return ep.split("/")[0]


class OSSDataloader(BaseDataLoader):
    """
    读取 segment jsonl + metadata_{subset}.jsonl，
    通过 MinIO 预签名 + ffmpeg 裁剪音频，产出音频样本。

    需要在 config 中提供：
      - meta_dir: metadata_{subset}.jsonl 所在目录
      - endpoint: MinIO endpoint（不带 http(s)）
      - access_key / secret_key
      - secure: 是否使用 HTTPS（bool，可选，默认 False）
      - presign_secs: 预签名有效期，默认 600
      - margin: 解码前向回探缓冲，默认 30.0s

    输出的每个样本包含：
      - audio: np.ndarray float32 单声道 24kHz 样本
      - sample_rate: 24000
      - start, end, duration, subset_name, metadata__id, object_key, bucket
    """

    SRC_PREFIX = "s3://archive-oss/nginx/data"
    DST_PREFIX = "qz_oss://embodied-multimodality-datasets/speech/datasets/Podcast"
    DST_BUCKET = "embodied-multimodality-datasets"

    def __init__(self, data_path: str, batch_size: int = 1, shuffle: bool = False, **kwargs):
        # 延迟导入以避免对未使用场景的依赖
        try:
            import ffmpeg  # noqa: F401
            import numpy  # noqa: F401
            from minio import Minio  # noqa: F401
        except Exception as e:
            logger.warning(f"依赖缺失：{e}. 需要安装 ffmpeg-python、numpy、minio 并确保系统有 ffmpeg/ffprobe")

        self.meta_dir: str = kwargs.get("meta_dir")
        self.endpoint: str = _sanitize_endpoint(kwargs.get("endpoint", ""))
        self.access_key: str = kwargs.get("access_key", "")
        self.secret_key: str = kwargs.get("secret_key", "")
        self.secure: bool = bool(kwargs.get("secure", False))
        self.presign_secs: int = int(kwargs.get("presign_secs", 600))
        self.margin: float = float(kwargs.get("margin", 30.0))
        # 输出给下游处理器的采样率，默认 16k 以适配 process_audio_info
        self.output_sample_rate: int = int(kwargs.get("output_sample_rate", 16000))
        # 流式读取，避免一次性载入巨大 JSONL
        self.streaming: bool = bool(kwargs.get("streaming", True))
        self._segment_files: List[str] = []
        self._num_items: Optional[int] = None

        if not self.meta_dir:
            raise ValueError("OSSDataloader 需要提供 meta_dir")
        if not self.endpoint or not self.access_key or not self.secret_key:
            raise ValueError("OSSDataloader 需要提供 endpoint/access_key/secret_key")

        # 缓存：subset -> { oid -> meta_obj }
        self._meta_cache: dict[str, dict[str, dict]] = {}

        # MinIO 客户端
        from minio import Minio
        self._minio_client = Minio(self.endpoint, access_key=self.access_key, secret_key=self.secret_key, secure=self.secure)

        super().__init__(data_path, batch_size, shuffle, **kwargs)

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        self.data = []

        def _gather_candidates(dir_or_file: str) -> List[str]:
            if os.path.isdir(dir_or_file):
                candidates = [
                    os.path.join(dir_or_file, name)
                    for name in os.listdir(dir_or_file)
                    if name.startswith("segment") and name.endswith(".jsonl")
                ]
                candidates.sort()
                if not candidates:
                    logger.warning(f"目录 {dir_or_file} 下未发现 segment*.jsonl")
                return candidates
            else:
                return [dir_or_file]

        if self.streaming:
            # 仅收集文件列表，不把行载入内存
            self._segment_files = _gather_candidates(self.data_path)
            logger.info(f"Streaming mode: {len(self._segment_files)} segment file(s) registered")
        else:
            # 兼容：一次性加载到内存（不推荐大文件）
            for fp in _gather_candidates(self.data_path):
                with open(fp, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if (
                            "start" not in item or "end" not in item or "subset_name" not in item or
                            "metadata__id" not in item or not isinstance(item["metadata__id"], dict) or
                            "$oid" not in item["metadata__id"]
                        ):
                            logger.warning(f"缺少必要字段，跳过：{item}")
                            continue
                        self.data.append(item)
            logger.info(f"Loaded {len(self.data)} segment items (non-streaming)")

    @staticmethod
    def _replace_prefix_path(src_path: str) -> str:
        if not src_path.startswith(OSSDataloader.SRC_PREFIX):
            raise ValueError(f"path前缀不匹配，期望以 {OSSDataloader.SRC_PREFIX} 开头，实际: {src_path}")
        return OSSDataloader.DST_PREFIX + src_path[len(OSSDataloader.SRC_PREFIX):]

    @staticmethod
    def _qz_to_bucket_and_key(qz_uri: str) -> Tuple[str, str]:
        if not qz_uri.startswith("qz_oss://"):
            raise ValueError(f"不支持的URI: {qz_uri}")
        rest = qz_uri[len("qz_oss://"):]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"无效的qz_oss URI: {qz_uri}")
        bucket, key = parts[0], parts[1]
        return bucket, key

    def _presign(self, bucket: str, object_name: str) -> str:
        return self._minio_client.get_presigned_url("GET", bucket, object_name, expires=timedelta(seconds=self.presign_secs))

    @staticmethod
    def _probe_realstart(url: str, rough: float, stream: str = "a:0", rw_timeout: str = "30M") -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", stream,
            "-read_intervals", f"{rough}%+#1",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "default=nw=1:nk=1",
            "-rw_timeout", rw_timeout,
            url,
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=True)
            out = p.stdout.strip()
            return float(out.splitlines()[0]) if out else rough
        except Exception:
            logger.info("ffprobe错误")
            return rough

    def _decode_segment(self, url: str, ss: float, t: float) -> "np.ndarray":
        import ffmpeg
        import numpy as np
        sr = 24000
        rough = max(ss - self.margin, 0.0)
        realstart = self._probe_realstart(url, rough)
        start_smp = max(int(round((ss - realstart) * sr)), 0)
        end_smp = start_smp + int(round(t * sr))
        try:
            out = (
                ffmpeg
                .input(url, ss=f"{rough:.9f}", seekable=1, rw_timeout="30M")
                .filter("aresample", sr)
                .filter('atrim', start_sample=start_smp, end_sample=end_smp)
                .filter('asetpts', 'PTS-STARTPTS')
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=sr)
                .global_args('-loglevel', 'error', '-probesize', '1M', '-analyzeduration', '5M')
                .run(capture_stdout=True, capture_stderr=True)[0]
            )
        except ffmpeg.Error as e:
            raise RuntimeError(e.stderr.decode('utf-8', errors='ignore'))
        return np.frombuffer(out, dtype=np.float32)

    def _resample_if_needed(self, audio_24k: "np.ndarray") -> "np.ndarray":
        """将 24k 音频按需重采样为 output_sample_rate（默认 16k）。"""
        import numpy as np
        target_sr = self.output_sample_rate
        if target_sr == 24000:
            return audio_24k
        try:
            import librosa
            audio_16k = librosa.resample(audio_24k, orig_sr=24000, target_sr=target_sr)
            return audio_16k.astype(np.float32, copy=False)
        except Exception as e:
            # 若无 librosa，退回原始 24k 以不中断流程
            logger.warning(f"重采样到 {target_sr} 失败或未安装librosa，继续使用24k: {e}")
            return audio_24k

    def _load_meta_subset(self, subset: str) -> dict:
        if subset in self._meta_cache:
            return self._meta_cache[subset]
        meta_file = os.path.join(self.meta_dir, f"metadata_{subset}.jsonl")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"找不到 {meta_file}")
        idx: dict[str, dict] = {}
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                oid = (((obj.get("_id") or {}).get("$oid")) or None)
                if oid:
                    idx[oid] = obj
        self._meta_cache[subset] = idx
        return idx

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        if self.streaming:
            if self.shuffle:
                logger.warning("Streaming 模式不支持打乱，已忽略 shuffle=True")
            batch: List[Dict[str, Any]] = []
            for fp in self._segment_files:
                with open(fp, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            seg = json.loads(line)
                        except Exception:
                            continue
                        if (
                            "start" not in seg or "end" not in seg or "subset_name" not in seg or
                            "metadata__id" not in seg or not isinstance(seg["metadata__id"], dict) or
                            "$oid" not in seg["metadata__id"]
                        ):
                            continue

                        start = float(seg["start"]) 
                        end = float(seg["end"]) 
                        duration = max(end - start, 0.0)
                        subset = str(seg["subset_name"]) 
                        meta_oid = ((seg.get("metadata__id") or {}).get("$oid"))

                        try:
                            subset_index = self._load_meta_subset(subset)
                            meta = subset_index.get(meta_oid)
                            if meta is None:
                                logger.warning(f"subset {subset} 未找到 oid {meta_oid}, 跳过")
                                continue
                            src_path = meta.get("path")
                            if not src_path:
                                logger.warning(f"meta 缺少 path 字段，跳过：{meta}")
                                continue
                            qz_uri = self._replace_prefix_path(src_path)
                            _bucket_from_uri, object_key = self._qz_to_bucket_and_key(qz_uri)
                            bucket = self.DST_BUCKET
                            url = self._presign(bucket, object_key)
                            audio_24k = self._decode_segment(url, ss=start, t=duration)
                            audio = self._resample_if_needed(audio_24k)

                            sample = {
                                "audio": audio,
                                "sample_rate": self.output_sample_rate if self.output_sample_rate else 24000,
                                "start": start,
                                "end": end,
                                "duration": duration,
                                "subset_name": subset,
                                "metadata__id": meta_oid,
                                "segment_id": seg.get("id"),
                                "bucket": bucket,
                                "object_key": object_key,
                            }
                            batch.append(sample)
                        except Exception as e:
                            logger.warning(f"OSSDataloader 流式样本处理失败 文件={fp}: {e}")
                            continue

                        if len(batch) == self.batch_size:
                            yield batch
                            batch = []

            if batch:
                yield batch
            return

        # 非流式：使用已加载的 self.data
        if self.shuffle:
            import random
            indices = list(range(len(self.data)))
            random.shuffle(indices)
        else:
            indices = list(range(len(self.data)))

        batch: List[Dict[str, Any]] = []
        for idx in indices:
            seg = self.data[idx]
            start = float(seg["start"]) 
            end = float(seg["end"]) 
            duration = max(end - start, 0.0)
            subset = str(seg["subset_name"]) 
            meta_oid = ((seg.get("metadata__id") or {}).get("$oid"))

            try:
                subset_index = self._load_meta_subset(subset)
                meta = subset_index.get(meta_oid)
                if meta is None:
                    logger.warning(f"subset {subset} 未找到 oid {meta_oid}, 跳过")
                    continue
                src_path = meta.get("path")
                if not src_path:
                    logger.warning(f"meta 缺少 path 字段，跳过：{meta}")
                    continue
                qz_uri = self._replace_prefix_path(src_path)
                _bucket_from_uri, object_key = self._qz_to_bucket_and_key(qz_uri)
                bucket = self.DST_BUCKET
                url = self._presign(bucket, object_key)
                audio_24k = self._decode_segment(url, ss=start, t=duration)
                audio = self._resample_if_needed(audio_24k)

                sample = {
                    "audio": audio,
                    "sample_rate": self.output_sample_rate if self.output_sample_rate else 24000,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "subset_name": subset,
                    "metadata__id": meta_oid,
                    "segment_id": seg.get("id"),
                    "bucket": bucket,
                    "object_key": object_key,
                }
                batch.append(sample)
            except Exception as e:
                logger.warning(f"OSSDataloader 样本处理失败 idx={idx}: {e}")
                continue

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def __len__(self) -> int:
        if not self.streaming:
            return len(self.data)
        # 流式模式：懒计数（可能较慢），仅在用户调用 len() 时执行一次
        if self._num_items is not None:
            return self._num_items
        total = 0
        for fp in self._segment_files:
            try:
                with open(fp, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if (
                            "start" in item and "end" in item and "subset_name" in item and
                            "metadata__id" in item and isinstance(item["metadata__id"], dict) and
                            "$oid" in item["metadata__id"]
                        ):    
                            total += 1
            except Exception:
                continue
        self._num_items = total
        return total


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
