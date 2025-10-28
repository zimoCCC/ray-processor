import os
import ray
from model_worker import create_qwen3omni_worker
from dataloader import OSSDataloader

# 建议的环境变量（先在进程内设置，若是多进程/容器也可放到外部）
os.environ["VLLM_USE_V1"] = "0"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
# 如需更强势排障：os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

ray.init()

worker = create_qwen3omni_worker(
    model_name="Qwen3-Omni-30B-A3B-Instruct",
    model_path="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Qwen3-Omni-30B-A3B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.97,          # 提高给KV的空间
    max_model_len=8192,                   # 降低KV尺寸
    max_num_seqs=1,                       # 降并发
    limit_mm_per_prompt={'image': 0, 'video': 0, 'audio': 3},  # 仅音频
    enforce_eager=True,                   # 先走eager模式
    max_tokens=128, 
)
print("create success")

# 从环境读取 OSSDataloader 配置，便于不改代码直接运行
segment_dir = os.environ.get("SEGMENT_DIR", "/inspire/ssd/project/embodied-multimodality/public/hfchen/meta/audio_h5")
meta_dir = os.environ.get("META_DIR", "/inspire/ssd/project/embodied-multimodality/public/hfchen/meta/audio_h5")
endpoint = os.environ.get("OSS_ENDPOINT", "oss.sii.shaipower.online:8009")
access_key = os.environ.get("OSS_ACCESS_KEY", "KDHLDKB84RDW4VE7P5KI")
secret_key = os.environ.get("OSS_SECRET_KEY", "oWPvR7UJqkLirm36uTgHqUbKGe8Hbk30BvK5PpVc")
secure = os.environ.get("OSS_SECURE", "false").lower() in ("1", "true", "yes")

loader = OSSDataloader(
    data_path=segment_dir,          # 目录：自动加载 segment*.jsonl
    batch_size=4,
    shuffle=False,
    meta_dir=meta_dir,              # 目录：包含 metadata_XX.jsonl
    endpoint=endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=secure,
    presign_secs=600,
    margin=30.0,
)

print("load success")

for batch in loader:
    future = worker.generate_batch.remote(batch)
    try:
        res = ray.get(future, timeout=30.0)   # 对单批设置超时
        print(res)
    except ray.exceptions.GetTimeoutError:
        # 取消这条卡住的任务，继续下一批
        try:
            ray.cancel(future, force=True)
        except Exception:
            pass
        print("Timeout on a batch; skipped.")
        continue