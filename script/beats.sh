#!/bin/bash

# 分布式推理启动脚本
# 使用方法: ./run_inference.sh
# 依赖环境变量: RANK 或 NODE_RANK

# 解析RANK（兼容RANK和NODE_RANK）
RANK_ID="${RANK:-$NODE_RANK}"

# 解析OFFSET（可选，默认为0）
OFFSET="${OFFSET:-0}"

# 检查环境变量
if [ -z "$RANK_ID" ]; then
    echo "Error: RANK or NODE_RANK environment variable is not set"
    echo "Please set RANK or NODE_RANK (0-15) before running this script"
    echo "Example: export RANK=0 && ./run_inference.sh"
    echo "Optional: export OFFSET=6 to process shard06 when RANK=0"
    exit 1
fi
DATA_PATH="/inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/ray_inference_framework/data/split"
WORKDIR="./workdir2"

# 创建工作目录
mkdir -p "$WORKDIR"

# 应用偏移到RANK（使后续输出与目录命名一致）
ORIGINAL_RANK="$RANK_ID"
RANK_ID=$((RANK_ID + OFFSET))

# 验证偏移后的RANK_ID范围
if [ $RANK_ID -lt 0 ] || [ $RANK_ID -gt 15 ]; then
    echo "Error: effective RANK (RANK+OFFSET) must be between 0 and 15, got: $RANK_ID (original=$ORIGINAL_RANK, offset=$OFFSET)"
    exit 1
fi

# 构造shard文件名
SHARD_FILE="shard$(printf "%02d" $RANK_ID).jsonl"
SHARD_PATH="$DATA_PATH/$SHARD_FILE"

# 检查shard文件是否存在
if [ ! -f "$SHARD_PATH" ]; then
    echo "Error: Shard file not found: $SHARD_PATH"
    exit 1
fi

# 设置输出目录
OUTPUT_DIR="$WORKDIR/${RANK_ID}_data"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 模型路径
MODEL_PATH="/inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/cargoflow/src/cargoflow/llms/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Starting inference for RANK=$RANK_ID (original=$ORIGINAL_RANK, offset=$OFFSET)"
echo "Data file: $SHARD_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_PATH"

# 运行推理
cd /inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/ray_inference_framework

# 确保输出目录存在（防止竞争条件）
mkdir -p "$OUTPUT_DIR"

python ray_inference.py \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/process.log" 2>&1

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "Inference completed successfully for RANK=$RANK_ID (processed $SHARD_FILE)"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Inference failed for RANK=$RANK_ID ($SHARD_FILE)"
    echo "Check log file: $OUTPUT_DIR/process.log"
    exit 1
fi
