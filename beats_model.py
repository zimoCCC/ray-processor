# -*- coding: utf-8 -*-
"""
BEATs模型实现
"""
import torch
import torchaudio
import numpy as np
from basemodel import BaseModel
import logging

logger = logging.getLogger(__name__)


class BEATsModel(BaseModel):
    """BEATs音频分类模型"""
    
    def __init__(self, model_name, device="cuda", model_path=None, **kwargs):
        """
        初始化BEATs模型
        
        Args:
            model_name: 模型名称
            device: 设备
            model_path: 模型文件路径
            **kwargs: 其他参数
        """
        self.model_path = model_path
        super().__init__(model_name, device, **kwargs)
    
    def _load_model(self):
        """加载BEATs模型"""
        from BEATs import BEATs, BEATsConfig
        
        # 加载checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 创建模型
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        # 保存标签字典
        self.label_dict = checkpoint['label_dict']
        
        logger.info(f"BEATs model loaded from {self.model_path}")
    
    def load_audio(self, audio_path, target_sample_rate=16000):
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            target_sample_rate: 目标采样率
            
        Returns:
            audio_tensor: 音频张量
        """
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到目标采样率
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)  # 移除batch维度
    
    def generate(self, inputs, top_k=5, **kwargs):
        """
        对音频进行分类推理
        
        Args:
            inputs: 音频文件路径列表或单个路径
            top_k: 返回top-k预测结果
            **kwargs: 其他参数
            
        Returns:
            分类结果
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        results = []
        for audio_path in inputs:
            # 加载音频
            audio_tensor = self.load_audio(audio_path)
            
            # 添加batch维度
            audio_input = audio_tensor.unsqueeze(0)  # [1, length]
            
            # 创建padding mask (False表示有效位置)
            padding_mask = torch.zeros(1, audio_input.shape[1]).bool().to(self.device)
            
            # 推理
            with torch.no_grad():
                probs = self.model.extract_features(audio_input.to(self.device), padding_mask=padding_mask)[0]
            
            # 获取top-k结果
            top_k_probs, top_k_indices = probs.topk(k=top_k)
            
            # 转换为标签
            top_k_labels = [self.label_dict[idx.item()] for idx in top_k_indices[0]]
            top_k_prob_values = top_k_probs[0].cpu().numpy().tolist()
            
            # 构建结果
            result = {
                "audio_path": audio_path,
                "predictions": [
                    {"label": label, "probability": float(prob)}
                    for label, prob in zip(top_k_labels, top_k_prob_values)
                ]
            }
            
            results.append(result)
        
        return results if len(results) > 1 else results[0]
