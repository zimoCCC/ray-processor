# -*- coding: utf-8 -*-
"""
BEATs模型使用示例
"""
import json
from ray_inference import run_inference, create_sample_data

def create_audio_sample_data(data_path, num_samples=10):
    """创建音频示例数据"""
    sample_data = []
    for i in range(num_samples):
        sample = {
            "id": f"audio_{i}",
            "audio_path": f"/path/to/audio_{i}.wav",  # 替换为实际音频路径
            "metadata": {
                "duration": 10.0 + i * 0.5,
                "sample_rate": 16000,
                "format": "wav"
            }
        }
        sample_data.append(sample)
    
    with open(data_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created {num_samples} audio samples at {data_path}")

if __name__ == "__main__":
    # 配置参数
    model_path = "/path/to/your/beats_model.pt"  # 替换为实际模型路径
    data_path = "audio_data.jsonl"
    output_path = "beats_results.jsonl"
    
    # 创建示例音频数据
    create_audio_sample_data(data_path, 10)
    
    # 运行BEATs推理
    print("Starting BEATs inference...")
    run_inference(
        data_path=data_path,
        output_path=output_path,
        num_gpus=2,  # 使用2个GPU
        batch_size=2,  # 音频处理建议小批次
        model_path=model_path
    )
    
    print("BEATs inference completed!")
    print(f"Results saved to: {output_path}")
