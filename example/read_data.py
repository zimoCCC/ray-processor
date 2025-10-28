#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
from contextlib import nullcontext
from datetime import timedelta
from urllib.parse import urlparse

import ffmpeg
import numpy as np
import wave
from minio import Minio

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# 原始路径前缀与目标映射
SRC_PREFIX = "s3://archive-oss/nginx/data"
DST_PREFIX = "qz_oss://embodied-multimodality-datasets/speech/datasets/Podcast"
DST_BUCKET = "embodied-multimodality-datasets"

def sanitize_endpoint(ep: str) -> str:
    ep = ep.strip()
    if ep.startswith("http://"):
        ep = ep[len("http://"):]
    elif ep.startswith("https://"):
        ep = ep[len("https://"):]
    return ep.split("/")[0]

def probe_realstart(url: str, rough: float, stream="a:0", rw_timeout="30M") -> float:
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
        logging.info("ffprobe错误")
        return rough

def ffmpeg_blocking_decode(url: str, ss: float, t: float, margin: float = 30.0) -> np.ndarray:
    sr = 24000
    rough = max(ss - margin, 0.0)
    realstart = probe_realstart(url, rough)
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

def save_wav_int16(path: str, samples_f32: np.ndarray, sample_rate: int = 24000):
    clipped = np.clip(samples_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

def replace_prefix_path(src_path: str) -> str:
    if not src_path.startswith(SRC_PREFIX):
        raise ValueError(f"path前缀不匹配，期望以 {SRC_PREFIX} 开头，实际: {src_path}")
    return DST_PREFIX + src_path[len(SRC_PREFIX):]

def qz_to_bucket_and_key(qz_uri: str) -> tuple[str, str]:
    # qz_oss://bucket/key...
    if not qz_uri.startswith("qz_oss://"):
        raise ValueError(f"不支持的URI: {qz_uri}")
    rest = qz_uri[len("qz_oss://"):]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"无效的qz_oss URI: {qz_uri}")
    bucket, key = parts[0], parts[1]
    return bucket, key

def presign_minio_url(endpoint: str, access_key: str, secret_key: str, bucket: str, object_name: str, expires_sec: int = 600, secure: bool = False) -> str:
    client = Minio(sanitize_endpoint(endpoint), access_key=access_key, secret_key=secret_key, secure=secure)
    return client.get_presigned_url("GET", bucket, object_name, expires=timedelta(seconds=expires_sec))

def read_jsonl_line(path: str, idx: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"{path} 不足 {idx+1} 行")

def find_meta_by_id(meta_jsonl_path: str, oid: str) -> dict | None:
    with open(meta_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            meta_id = (((obj.get("_id") or {}).get("$oid")) or None)
            if meta_id == oid:
                return obj
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment_jsonl", required=True)
    ap.add_argument("--segment_index", type=int, default=0)
    ap.add_argument("--meta_dir", required=True, help="包含 metadata_{subset}.jsonl 的目录")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--access_key", required=True)
    ap.add_argument("--secret_key", required=True)
    ap.add_argument("--secure", action="store_true")
    ap.add_argument("--presign_secs", type=int, default=600)
    ap.add_argument("--out", default="segment.wav")
    args = ap.parse_args()

    seg = read_jsonl_line(args.segment_jsonl, args.segment_index)

    # 从 segment 取字段
    print(seg)
    start = float(seg["start"])
    end = float(seg["end"])
    duration = end - start
    subset = str(seg["subset_name"])
    meta_oid = ((seg.get("metadata__id") or {}).get("$oid"))
    if not meta_oid:
        raise ValueError("segment 缺少 metadata__id.$oid")

    # 打开对应 metadata_{subset}.jsonl 查找同 oid
    meta_file = os.path.join(args.meta_dir, f"metadata_{subset}.jsonl")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"找不到 {meta_file}")
    meta = find_meta_by_id(meta_file, meta_oid)
    if not meta:
        raise ValueError(f"{meta_file} 未找到 _id.$oid = {meta_oid}")

    src_path = meta["path"]  # e.g. s3://archive-oss/nginx/data/...
    qz_uri = replace_prefix_path(src_path)  # qz_oss://embodied-multimodality-datasets/speech/datasets/Podcast/...
    bucket_from_uri, object_key = qz_to_bucket_and_key(qz_uri)

    # 按你的要求 bucket 固定为 embodied-multimodality-datasets
    bucket = DST_BUCKET
    if bucket_from_uri != bucket:
        logging.warning(f"qz uri bucket为 {bucket_from_uri}，按规则使用 {bucket}")

    # presign 后用 ffmpeg 裁剪
    url = presign_minio_url(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket=bucket,
        object_name=object_key,
        expires_sec=args.presign_secs,
        secure=args.secure,
    )
    logging.info("已生成 presigned URL")

    samples = ffmpeg_blocking_decode(url, ss=start, t=duration, margin=30.0)
    logging.info(f"decoded {samples.shape[0]} samples @24kHz")

    out_path = os.path.abspath(args.out)
    save_wav_int16(out_path, samples, sample_rate=24000)
    logging.info(f"保存到 {out_path}")

if __name__ == "__main__":
    main()