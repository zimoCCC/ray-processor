import torch
import torchaudio
from BEATs import BEATs, BEATsConfig

# load the fine-tuned checkpoints
checkpoint = torch.load('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
test_wav, sr = torchaudio.load(
    '/inspire/hdd/project/embodied-multimodality/public/hfchen/cargoflow/cargoflow/work_dirs/debug/audios/7_0.wav'
)

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()


# predict the classification probability of each class
# audio_input_16khz = torch.randn(3, 10000)
# padding_mask = torch.zeros(3, 10000).bool()
padding_mask = torch.zeros(test_wav.shape[0], test_wav.shape[1], dtype=torch.bool)
probs = BEATs_model.extract_features(test_wav, padding_mask=padding_mask)[0]
print(test_wav.shape)
for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
    top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
    print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')
