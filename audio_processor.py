import os
import json
import numpy as np
from scipy.io.wavfile import read
from dft import analyze_audio_chunk
from chord_collector import major_chords

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
        
        # Normalize audio to [-1, 1] ranged
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Take a reasonable chunk for analysis (4096 samples = ~0.1 seconds at 44kHz)
        max_samples = min(len(audio_data), 4096)
        audio_chunk = audio_data[:max_samples]
        
        print(f"Processing {wav_filename} ({len(audio_data)} samples → {max_samples} samples)...")
        
        # Use the DFT module to analyze the audio chunk
        print(f"Applying custom DFT analysis...")
        dft_results = analyze_audio_chunk(audio_chunk, sample_rate, chunk_size=512)
        
        # Extract results from DFT analysis
        peak_frequency = dft_results["peak_frequency_hz"]
        peak_magnitude = dft_results["peak_magnitude"]
        top_frequencies = dft_results["top_5_frequencies"]
        
        # Prepare output data combining file info with DFT results
        analysis_data = {
            "filename": wav_filename,
            "sample_rate": int(sample_rate),
            "audio_length_seconds": float(len(audio_data) / sample_rate),
            "peak_frequency_hz": float(peak_frequency),
            "peak_magnitude": float(peak_magnitude),
            "top_5_frequencies": top_frequencies,
            **dft_results  # Include all DFT analysis results
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
    print("Starting audio processing for all recorded chords...")
    print(f"Will process {len(major_chords)} chords: {', '.join(major_chords)}")
    process_all_chords(major_chords)
    print("\nAudio processing complete!")
    