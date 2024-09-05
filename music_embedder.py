import torch
import librosa
from torch import autocast
from contextlib import nullcontext

from models.dymn.model import get_model as get_dymn
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH


model_name = "dymn10_as"
print("Model:", model_name)
device_="cpu"
strides=[2, 2, 2, 2]
n_mels=128
sample_rate=32000
window_size=800
hop_size=320

# load pre-trained model
if model_name.startswith("dymn"):
    model = get_dymn(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=strides)
else:
    model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,
                                  strides=strides)

device = torch.device('cuda') if device_ == "cuda" and torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.eval()
# model to preprocess waveform into mel spectrograms
mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
mel.to(device)
mel.eval()

def audio_embed(audio_path, head_type="mlp"):
    """
    Running Inference on an audio clip.
    """

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    with torch.no_grad(), autocast(device_type=device.type) if device_ == "cuda" else nullcontext():
        spec = mel(waveform)
        _preds, features = model(spec.unsqueeze(0))
    
    return features.squeeze(0).detach().numpy()
