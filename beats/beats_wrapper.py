import torch
import torchaudio
from typing import List, Dict, Optional, Union
from BEATs import BEATs as RawBEATs, BEATsConfig

class BEATsWrapper:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(ckpt["cfg"])
        model = RawBEATs(cfg)
        model.load_state_dict(ckpt["model"])
        self.model = model.eval()
        self.id2mid = ckpt["label_dict"]  # idx -> MID

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def _to_mono16k(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # wav: [C, T] or [T]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]
        if wav.size(0) > 1:  # 多声道转单声道
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        return wav.float()  # [1, T]

    def _pad_batch(self, wavs: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
        max_len = int(lengths.max())
        batch = torch.zeros(len(wavs), max_len, dtype=torch.float32)
        for i, w in enumerate(wavs):
            batch[i, : w.shape[-1]] = w
        padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)  # True=pad
        return batch, padding_mask

    @torch.no_grad()
    def generate(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],  # 单条 [T]/[C,T] 或 列表
        sample_rate: int,
        topk: int = 5,
        return_names: bool = False,  # 若提供了 mid2name，可返回可读名称
    ) -> List[Dict[str, List]]:
        # 统一到列表
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        # 预处理到 16k 单声道
        wavs = [self._to_mono16k(w, sample_rate) for w in inputs]  # 每个 [1, T]
        wavs = [w.squeeze(0) for w in wavs]  # -> [T]
        batch, padding_mask = self._pad_batch(wavs)

        batch = batch.to(self.device)
        padding_mask = padding_mask.to(self.device)

        # 提取类别概率（部分权重返回在 output[0]）
        probs = self.model.extract_features(batch, padding_mask=padding_mask)[0]  # [B, num_classes]

        top_vals, top_idx = torch.topk(probs, k=topk, dim=-1)  # [B, k]
        results = []
        for i in range(top_vals.size(0)):
            idx_list = top_idx[i].tolist()
            prob_list = top_vals[i].tolist()
            mids = [self.id2mid[j] for j in idx_list]
            if return_names and self.mid2name is not None:
                labels = [self.mid2name.get(mid, mid) for mid in mids]
            else:
                labels = mids
            results.append({"labels": labels, "probs": prob_list})
        return results
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/cargoflow/src/cargoflow/llms/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt", help="Path to BEATs checkpoint .pt")
    parser.add_argument("--wav", type=str, default="/inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/cargoflow/work_dirs/debug/audios/2_0.wav", help="Path to .wav file")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    # 初始化包装器
    wrapper = BEATsWrapper(args.ckpt)

    # 读取音频
    wav, sr = torchaudio.load(args.wav)  # wav: [C, T], sr: int

    # 调用 generate
    outputs = wrapper.generate(wav, sample_rate=sr, topk=args.topk, return_names=False)

    # 打印结果
    for i, out in enumerate(outputs):
        probs_fmt = [f"{p:.2%}" for p in out["probs"]]
        print(f"Top{args.topk} of sample {i}: {out['labels']} with probs {probs_fmt}")