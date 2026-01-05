import io
import numpy as np
import soundfile as sf
from PIL import Image

TARGET_SR = 22050
MAX_SECONDS = 5

def audio_to_spectrogram(audio_bytes):
    print("➡️ audio_to_spectrogram: start")

    # Decode WAV
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    print("➡️ audio decoded:", y.shape, "sr:", sr)

    # Mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Trim to max duration
    y = y[: sr * MAX_SECONDS]

    # Manual downsample (safe)
    if sr != TARGET_SR:
        factor = sr // TARGET_SR
        y = y[::factor]
        sr = TARGET_SR
        print("➡️ audio downsampled manually to", sr)

    # 🔑 VERY LIGHT FEATURE: frame energy map
    frame_size = 512
    hop = 256

    frames = []
    for i in range(0, len(y) - frame_size, hop):
        frame = y[i:i+frame_size]
        energy = np.sum(frame ** 2)
        frames.append(energy)

    spec = np.array(frames)

    # Normalize
    spec -= spec.min()
    spec /= (spec.max() + 1e-6)
    spec *= 255.0

    # Convert to image
    img = Image.fromarray(spec.astype(np.uint8)).convert("L")
    img = img.resize((224, 224))

    print("➡️ spectrogram image ready")
    return img
