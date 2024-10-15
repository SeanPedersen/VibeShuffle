import torch
import torch.jit
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torchaudio.transforms as T
import numpy as np
import os


model_name = "m-a-p/MERT-v1-330M"  # "m-a-p/MERT-v1-95M" "m-a-p/MERT-v1-330M"
print("Model:", model_name)

# load pre-trained model
# loading our model weights
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)

torch.set_num_threads(os.cpu_count())
if torch.backends.mps.is_available():
    # OOM errors out and not really faster -> shitty accelerator
    # Convert model to bfloat16
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")  # TODO: remove this hardcoded CPU device
print("Device:", device)
if str(device) == "mps":
    print("Converting model to bfloat16")
    model = model.to(torch.bfloat16)

model.to(device)
model.eval()

resample_rate = processor.sampling_rate


def select_audio_segments(input_audio, target_length=3776122):
    # Keep first quarter, half from middle and last quarter as representation sample
    total_length = len(input_audio)

    if total_length <= target_length:
        return input_audio

    quarter_length = total_length // 4
    mid_half_length = (target_length - 2 * quarter_length) // 2

    first_quarter = input_audio[:quarter_length]
    last_quarter = input_audio[-quarter_length:]

    mid_start = (total_length - mid_half_length) // 2
    mid_section = input_audio[mid_start : mid_start + mid_half_length]

    return np.concatenate([first_quarter, mid_section, last_quarter])


def audio_embed(audio_path):
    """
    Running Inference on an audio clip.
    """

    # Load MP3 file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    resample_rate = processor.sampling_rate
    # Make sure the sample_rate is aligned
    if resample_rate != sample_rate:
        resampler = T.Resample(sample_rate, resample_rate)
        waveform = resampler(waveform)

    # Prepare input
    input_audio = waveform.squeeze().numpy()
    if str(device) == "mps":
        # MPS can not handle big audio files (OOM errors)
        input_audio = select_audio_segments(input_audio)
    inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
    if str(device) == "mps":
        inputs = {k: v.to(torch.bfloat16).to(device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    # Process through the model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Take a look at the output shape, there are 13 layers of representation
    # Each layer performs differently in different downstream tasks, you should choose empirically
    # Layer 7 should be good for genre classification
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()[6]
    # For utterance level classification tasks, you can simply reduce the representation in time
    time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
    return time_reduced_hidden_states.cpu().squeeze().detach().to(torch.float32).numpy()
