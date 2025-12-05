import os
import sounddevice as sd
from scipy.io.wavfile import write

duration = 5
sample_rate = 44100

def audio_recorder(chord=None, folder=None, filename=None):
    """
    Records audio and saves it to a specified location.
    
    Args:
        chord: The chord name (used for default filename if not provided)
        folder: The folder path where the file will be saved
        filename: Optional specific filename (if not provided, generates one)
    """
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Done recording!")
    
    if filename:
        filepath = os.path.join(folder, filename) if folder else filename
    else:
        filepath = os.path.join(folder, f"output_{chord}.wav") if folder else f"output_{chord}.wav"
    
    write(filepath, sample_rate, audio)
    print(f"Saved as {filepath}")
