import math
import numpy as np

def dft(signal):
    """
    Custom Discrete Fourier Transform implementation.
    
    Args:
        signal: Input audio signal (list or array of samples)
    
    Returns:
        List of magnitude values for each frequency bin
    """
    N = len(signal)
    result = []
    for k in range(N):
        real_part = 0
        imag_part = 0
        for n in range(N):
            angle = (2 * math.pi * k * n) / N
            real_part += signal[n] * math.cos(angle)
            imag_part += signal[n] * math.sin(angle)
        magnitude = math.sqrt(real_part**2 + imag_part**2)
        result.append(magnitude)
    return result

def analyze_audio_chunk(audio_data, sample_rate, chunk_size=512):
    """
    Analyze an audio chunk using DFT and extract frequency information.
    
    Args:
        audio_data: Input audio samples
        sample_rate: Audio sample rate (Hz)
        chunk_size: Number of samples to analyze (default: 512)
    
    Returns:
        Dictionary containing frequency analysis results
    """
    # Take optimal chunk size for processing
    optimal_samples = min(chunk_size, len(audio_data))
    dft_chunk = audio_data[:optimal_samples]
    
    # Apply custom DFT
    dft_magnitudes = dft(dft_chunk)
    
    # Calculate frequency axis (Hz) for the DFT chunk
    freqs = np.fft.fftfreq(len(dft_chunk), 1/sample_rate)[:len(dft_magnitudes)]
    
    # Find peak frequency
    peak_idx = np.argmax(dft_magnitudes)
    peak_frequency = freqs[peak_idx]
    peak_magnitude = dft_magnitudes[peak_idx]
    
    # Find top 5 frequencies
    top_indices = np.argsort(dft_magnitudes)[-5:][::-1]
    top_frequencies = [(float(freqs[i]), float(dft_magnitudes[i])) for i in top_indices]
    
    return {
        "processed_samples": int(optimal_samples),
        "processed_length_seconds": float(optimal_samples / sample_rate),
        "peak_frequency_hz": float(peak_frequency),
        "peak_magnitude": float(peak_magnitude),
        "top_5_frequencies": top_frequencies,
        "dft_magnitudes": [float(m) for m in dft_magnitudes],
        "frequencies_hz": [float(f) for f in freqs]
    }