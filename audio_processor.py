import os
import json
import numpy as np
from scipy.io.wavfile import read
from dft import dft

CHORD_RECORDINGS_PATH = "chord_recordings"
ANALYSIS_OUTPUT_PATH = "analysis"

def ensure_analysis_folder():
    """Create analysis folder if it doesn't exist."""
    if not os.path.exists(ANALYSIS_OUTPUT_PATH):
        os.makedirs(ANALYSIS_OUTPUT_PATH)

def process_chord_samples(chord_name):
    """
    Process all samples for a given chord and extract frequency features.
    
    Args:
        chord_name: Name of the chord (e.g., "C", "G#")
    
    Returns:
        True if processing successful, False otherwise
    """
    chord_folder = os.path.join(CHORD_RECORDINGS_PATH, chord_name)
    
    if not os.path.exists(chord_folder):
        print(f"Chord folder not found: {chord_folder}")
        return False
    
    # Create chord-specific analysis folder
    analysis_chord_folder = os.path.join(ANALYSIS_OUTPUT_PATH, chord_name)
    if not os.path.exists(analysis_chord_folder):
        os.makedirs(analysis_chord_folder)
    
    # Get all WAV files in the chord folder
    wav_files = [f for f in os.listdir(chord_folder) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in {chord_folder}")
        return False
    
    print(f"Processing {len(wav_files)} samples for chord: {chord_name}")
    
    for wav_file in sorted(wav_files):
        wav_path = os.path.join(chord_folder, wav_file)
        process_single_file(wav_path, wav_file, analysis_chord_folder)
    
    return True

def process_single_file(wav_path, wav_filename, output_folder):
    """
    Process a single WAV file: read audio, apply DFT, extract features.
    
    Args:
        wav_path: Full path to the WAV file
        wav_filename: Name of the WAV file
        output_folder: Where to save the JSON analysis
    """
    try:
        # Read WAV file
        sample_rate, audio_data = read(wav_path)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio to [-1, 1] range
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply DFT
        dft_magnitudes = dft(audio_data)
        
        # Calculate frequency axis (Hz)
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(dft_magnitudes)]
        freqs = np.abs(freqs)  # Take absolute values for positive frequencies
        
        # Find peak frequency
        peak_idx = np.argmax(dft_magnitudes)
        peak_frequency = freqs[peak_idx]
        peak_magnitude = dft_magnitudes[peak_idx]
        
        # Find top 5 frequencies
        top_indices = np.argsort(dft_magnitudes)[-5:][::-1]
        top_frequencies = [(float(freqs[i]), float(dft_magnitudes[i])) for i in top_indices]
        
        # Prepare output data
        analysis_data = {
            "filename": wav_filename,
            "sample_rate": int(sample_rate),
            "audio_length_seconds": float(len(audio_data) / sample_rate),
            "peak_frequency_hz": float(peak_frequency),
            "peak_magnitude": float(peak_magnitude),
            "top_5_frequencies": top_frequencies,
            "dft_magnitudes": [float(m) for m in dft_magnitudes],
            "frequencies_hz": [float(f) for f in freqs]
        }
        
        # Save as JSON
        json_filename = wav_filename.replace('.wav', '.json')
        json_path = os.path.join(output_folder, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"  ✓ {wav_filename} → Peak: {peak_frequency:.2f} Hz")
        
    except Exception as e:
        print(f"  ✗ Error processing {wav_filename}: {str(e)}")

def process_all_chords(chord_list):
    """
    Process all chords in the provided list.
    
    Args:
        chord_list: List of chord names to process
    """
    ensure_analysis_folder()
    
    successful = 0
    failed = 0
    
    for chord in chord_list:
        if process_chord_samples(chord):
            successful += 1
        else:
            failed += 1
    
    print(f"\n✓ Processed: {successful} chords")
    if failed > 0:
        print(f"✗ Failed: {failed} chords")

if __name__ == "__main__":
    # Example usage
    chords = ["A", "B", "C"]
    process_all_chords(chords)
