import os
import json
import numpy as np
from scipy.io.wavfile import read, write
from dft import analyze_audio_chunk
from audio_recorder import audio_recorder
import tempfile

ANALYSIS_PATH = "analysis"
CHORD_FOLDERS = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E"]  # Available analyzed chords

class ChordRecognizer:
    def __init__(self):
        self.chord_patterns = {}
        self.load_chord_patterns()
    
    def load_chord_patterns(self):
        """
        Load analyzed chord patterns from the analysis folder.
        Creates average frequency signatures for each chord.
        """
        print("Loading chord patterns...")
        
        for chord_name in CHORD_FOLDERS:
            chord_analysis_path = os.path.join(ANALYSIS_PATH, chord_name)
            
            if not os.path.exists(chord_analysis_path):
                print(f"  Warning: No analysis found for chord {chord_name}")
                continue
            
            # Load all analysis files for this chord
            json_files = [f for f in os.listdir(chord_analysis_path) if f.endswith('.json')]
            
            if not json_files:
                print(f"  Warning: No analysis files found for chord {chord_name}")
                continue
            
            chord_data = []
            for json_file in json_files:
                json_path = os.path.join(chord_analysis_path, json_file)
                
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    chord_data.append(data)
                except Exception as e:
                    print(f"  Error loading {json_file}: {e}")
            
            if chord_data:
                self.chord_patterns[chord_name] = self.create_chord_signature(chord_data)
                print(f"  ✓ Loaded {len(chord_data)} samples for chord {chord_name}")
        
        print(f"Successfully loaded patterns for {len(self.chord_patterns)} chords\n")
    
    def create_chord_signature(self, chord_data_list):
        """
        Create a representative signature for a chord based on multiple samples.
        
        Args:
            chord_data_list: List of analysis data dictionaries for the chord
            
        Returns:
            Dictionary containing the chord's frequency signature
        """
        # Extract key features from each sample
        peak_frequencies = []
        top_frequencies_all = []
        
        for data in chord_data_list:
            peak_frequencies.append(abs(data['peak_frequency_hz']))
            
            # Get the top 3 most significant frequencies (excluding negative mirrors)
            top_freqs = []
            for freq, magnitude in data['top_5_frequencies']:
                if freq > 0:  # Only positive frequencies
                    top_freqs.append((freq, magnitude))
            
            # Sort by magnitude and take top 3
            top_freqs.sort(key=lambda x: x[1], reverse=True)
            top_frequencies_all.append(top_freqs[:3])
        
        # Calculate average characteristics
        avg_peak_freq = np.mean(peak_frequencies)
        
        # Find the most common frequency patterns
        all_significant_freqs = []
        for top_freqs in top_frequencies_all:
            for freq, _ in top_freqs:
                all_significant_freqs.append(freq)
        
        # Create frequency bins and find most common ranges
        freq_bins = np.histogram(all_significant_freqs, bins=20)[0]
        
        return {
            'avg_peak_frequency': avg_peak_freq,
            'frequency_distribution': freq_bins,
            'sample_count': len(chord_data_list),
            'all_significant_frequencies': all_significant_freqs
        }
    
    def record_and_analyze_input(self):
        """
        Record audio input and analyze it for chord recognition.
        
        Returns:
            Dictionary containing the analysis results
        """
        print("Recording audio for chord recognition...")
        print("Play a chord on your guitar now!")
        
        # Create temporary file for recording
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            # Record audio (uses default 5 seconds from audio_recorder)
            audio_recorder("input_chord", filename=os.path.basename(temp_filename), 
                          folder=os.path.dirname(temp_filename))
            
            # Read the recorded audio
            sample_rate, audio_data = read(temp_filename)
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Analyze using your custom DFT
            analysis_results = analyze_audio_chunk(audio_data, sample_rate, chunk_size=512)
            
            print(f"✓ Recorded and analyzed {len(audio_data)} samples")
            return analysis_results
            
        except Exception as e:
            print(f"Error during recording/analysis: {e}")
            return None
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def calculate_chord_similarity(self, input_analysis, chord_signature):
        """
        Calculate similarity between input analysis and a chord signature.
        
        Args:
            input_analysis: Analysis results from recorded input
            chord_signature: Stored signature for a known chord
            
        Returns:
            Similarity score (higher = more similar)
        """
        if not input_analysis:
            return 0
        
        # Compare peak frequencies
        input_peak = abs(input_analysis['peak_frequency_hz'])
        signature_peak = chord_signature['avg_peak_frequency']
        
        # Calculate frequency similarity (closer = higher score)
        freq_diff = abs(input_peak - signature_peak)
        max_freq_diff = 1000  # Hz
        freq_similarity = max(0, 1 - (freq_diff / max_freq_diff))
        
        # Compare frequency distributions
        input_top_freqs = [abs(freq) for freq, mag in input_analysis['top_5_frequencies'][:3] if freq > 0]
        signature_freqs = chord_signature['all_significant_frequencies']
        
        # Count matches within tolerance
        matches = 0
        tolerance = 50  # Hz tolerance for frequency matching
        
        for input_freq in input_top_freqs:
            for sig_freq in signature_freqs:
                if abs(input_freq - sig_freq) <= tolerance:
                    matches += 1
                    break
        
        freq_pattern_similarity = matches / max(len(input_top_freqs), 1)
        
        # Combine similarities (weighted)
        total_similarity = (freq_similarity * 0.6) + (freq_pattern_similarity * 0.4)
        
        return total_similarity
    
    def recognize_chord(self):
        """
        Main chord recognition function.
        Records audio and identifies the most likely chord.
        """
        if not self.chord_patterns:
            print("Error: No chord patterns loaded!")
            return
        
        # Record and analyze input
        input_analysis = self.record_and_analyze_input()
        
        if not input_analysis:
            print("Failed to analyze input audio")
            return
        
        print(f"\nInput analysis:")
        print(f"  Peak frequency: {input_analysis['peak_frequency_hz']:.1f} Hz")
        print(f"  Top frequencies: {[f'{freq:.1f}Hz' for freq, _ in input_analysis['top_5_frequencies'][:3]]}")
        
        # Compare against all known chords
        similarities = {}
        
        for chord_name, signature in self.chord_patterns.items():
            similarity = self.calculate_chord_similarity(input_analysis, signature)
            similarities[chord_name] = similarity
        
        # Sort by similarity
        sorted_chords = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nChord Recognition Results:")
        print("=" * 40)
        
        for i, (chord, similarity) in enumerate(sorted_chords[:3]):
            confidence = similarity * 100
            print(f"{i+1}. {chord:>2} chord - {confidence:.1f}% confidence")
        
        # Announce the best match
        best_chord, best_similarity = sorted_chords[0]
        
        if best_similarity > 0.3:  # Confidence threshold
            print(f"\n🎸 Detected chord: {best_chord} (confidence: {best_similarity*100:.1f}%)")
        else:
            print(f"\n❓ Chord not clearly recognized. Best guess: {best_chord} (low confidence: {best_similarity*100:.1f}%)")
        
        return best_chord, best_similarity

def main():
    """
    Main function to run the chord recognizer.
    """
    print("🎸 Guitar Chord Recognizer")
    print("=" * 40)
    
    recognizer = ChordRecognizer()
    
    if not recognizer.chord_patterns:
        print("No chord patterns available. Please run audio_processor.py first to analyze your chord recordings.")
        return
    
    while True:
        print(f"\nAvailable commands:")
        print("  'r' or 'recognize' - Recognize a chord")
        print("  'q' or 'quit' - Exit")
        
        command = input("\nEnter command: ").strip().lower()
        
        if command in ['q', 'quit', 'exit']:
            print("Goodbye! 🎸")
            break
        elif command in ['r', 'recognize', '']:
            print()
            recognizer.recognize_chord()
        else:
            print("Unknown command. Try 'r' to recognize a chord or 'q' to quit.")

if __name__ == "__main__":
    main()
