"""
Chord recognition system using manual DFT with terminal interface
"""

import time
import sounddevice as sd
import json
import os
import numpy as np
from dft import ManualDFT
import matplotlib.pyplot as plt


class ChordRecognizer:
    """Main chord recognizer class"""
    
    def __init__(self, sample_rate=11000, duration=2.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.chord_library = {}
        self.library_file = "chord_library.json"
        self.dft = ManualDFT()
        
        # Load existing library if available
        self.load_library()
    
    def record_audio(self, message="Recording..."):
        """Record audio from microphone"""

        print("pausing for 1 second before recording...")
        time.sleep(1)  # Short pause before recording
    
        print(f"\n{message}")
        print(f"Recording for {self.duration} seconds...")
        
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("Recording complete!")
        
        # Convert to 1D array
        return audio.flatten()
    
    def extract_features(self, audio_signal):
        """
        Extract frequency features from audio using manual DFT
        Returns the top frequency peaks and their magnitudes
        """
        # Downsample to 0.5 seconds worth of data spread over the full recording
        target_duration = 0.5  # seconds
        target_samples = int(target_duration * self.sample_rate)
        
        # Calculate downsampling factor
        downsample_factor = len(audio_signal) // target_samples
        
        # Take every Nth sample to get 0.5 seconds of data from the full recording
        audio_downsampled = audio_signal[::downsample_factor][:target_samples]
        
        print(f"Processing {len(audio_downsampled)} samples from {len(audio_signal)} (every {downsample_factor}th sample)...")
        
        # Perform manual DFT on the downsampled signal
        frequencies = self.dft.dft(audio_downsampled)
        magnitudes = self.dft.get_magnitude_spectrum(frequencies)
        
        # Get frequency bins (use original sample rate for frequency calculation)
        freq_bins = [(i * self.sample_rate / len(audio_downsampled), magnitudes[i]) 
                     for i in range(len(magnitudes))]
        
        # Sort by magnitude and get top peaks
        freq_bins.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 10 frequency peaks (chords typically have 3-6 fundamental frequencies)
        top_peaks = freq_bins[:10]
        
        # Filter out very low frequencies (below 80 Hz, typical guitar range starts ~82 Hz)
        top_peaks = [(freq, mag) for freq, mag in top_peaks if freq > 80 and freq < 1500]
        
        return top_peaks

    def _compute_spectrum(self, signal_segment):
        """Compute frequency bins (Hz) and magnitude spectrum for a signal segment"""
        freqs_complex = self.dft.dft(signal_segment)
        magnitudes = self.dft.get_magnitude_spectrum(freqs_complex)
        N = len(signal_segment)
        freq_bins_hz = [i * self.sample_rate / N for i in range(len(magnitudes))]
        return freq_bins_hz, magnitudes

    def _plot_training_spectra(self, chord_name, audio_signal):
        """Plot two DFT magnitude spectra: downsampled spread vs contiguous 0.5s"""
        # Prepare signals
        target_duration = 0.5
        target_samples = int(target_duration * self.sample_rate)

        # Spread downsampling across full recording
        downsample_factor = max(1, len(audio_signal) // target_samples)
        spread_signal = audio_signal[::downsample_factor][:target_samples]

        # Contiguous 0.5s from start
        contiguous_signal = audio_signal[:target_samples]

        # Compute spectra
        freq_spread, mag_spread = self._compute_spectrum(spread_signal)
        freq_contig, mag_contig = self._compute_spectrum(contiguous_signal)

        # Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(freq_spread, mag_spread, color='tab:blue')
        plt.title(f"{chord_name} - Downsampled (spread)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("DFT Magnitude")
        plt.xlim(0, 2000)

        plt.subplot(1, 2, 2)
        plt.plot(freq_contig, mag_contig, color='tab:orange')
        plt.title(f"{chord_name} - Contiguous (first 0.5s)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("DFT Magnitude")
        plt.xlim(0, 2000)

        plt.tight_layout()

        # Ensure output directory exists
        os.makedirs("dft_plots", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("dft_plots", f"{chord_name}_{ts}.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved DFT plots to {out_path}")

        # Show the plot for immediate feedback
        plt.show()

    def train_chord(self, chord_name):
        """Record and store a chord in the library"""
        audio = self.record_audio(f"Recording chord: {chord_name}")
        features = self.extract_features(audio)

        self.chord_library[chord_name] = features
        print(f"\nChord '{chord_name}' features extracted:")
        print("Top frequencies (Hz) and magnitudes:")
        for freq, mag in features[:5]:
            print(f"  {freq:.1f} Hz: {mag:.2f}")

        # Plot two spectra after training
        self._plot_training_spectra(chord_name, audio)

        self.save_library()
    
    def compare_features(self, features1, features2):
        """
        Compare two feature sets and return similarity score
        Uses frequency matching with magnitude weighting
        """
        if not features1 or not features2:
            return 0.0
        
        score = 0.0
        tolerance = 50  # Hz tolerance for frequency matching
        
        # Compare each frequency in features1 with features2
        for freq1, mag1 in features1[:6]:  # Compare top 6 frequencies
            best_match = 0.0
            for freq2, mag2 in features2[:6]:
                if abs(freq1 - freq2) < tolerance:
                    # Calculate similarity based on frequency closeness and magnitude
                    freq_similarity = 1.0 - (abs(freq1 - freq2) / tolerance)
                    mag_similarity = min(mag1, mag2) / max(mag1, mag2) if max(mag1, mag2) > 0 else 0
                    match = freq_similarity * mag_similarity
                    best_match = max(best_match, match)
            
            score += best_match
        
        # Normalize score
        return score / 6.0
    
    def recognize_chord(self):
        """Record live audio and identify the chord"""
        if not self.chord_library:
            print("\nNo chords in library! Please train some chords first.")
            return None
        
        audio = self.record_audio("Recording for chord recognition...")
        features = self.extract_features(audio)
        
        print("\nDetected frequencies:")
        for freq, mag in features[:5]:
            print(f"  {freq:.1f} Hz: {mag:.2f}")
        
        # Compare with all stored chords
        best_match = None
        best_score = 0.0
        
        print("\nMatching against library...")
        for chord_name, stored_features in self.chord_library.items():
            score = self.compare_features(features, stored_features)
            print(f"  {chord_name}: {score:.2%} match")
            
            if score > best_score:
                best_score = score
                best_match = chord_name
        
        if best_score > 0.3:  # Threshold for valid match
            print(f"\n*** Recognized chord: {best_match} (confidence: {best_score:.2%}) ***")
            return best_match
        else:
            print(f"\n*** No confident match found (best: {best_score:.2%}) ***")
            return None
    
    def save_library(self):
        """Save chord library to file"""
        with open(self.library_file, 'w') as f:
            json.dump(self.chord_library, f, indent=2)
        print(f"Library saved to {self.library_file}")
    
    def load_library(self):
        """Load chord library from file"""
        if os.path.exists(self.library_file):
            with open(self.library_file, 'r') as f:
                self.chord_library = json.load(f)
            print(f"Loaded {len(self.chord_library)} chords from library")
    
    def list_chords(self):
        """List all chords in the library"""
        if not self.chord_library:
            print("\nNo chords in library yet.")
        else:
            print(f"\nChords in library ({len(self.chord_library)}):")
            for chord_name in self.chord_library.keys():
                print(f"  - {chord_name}")
    
    def detect_chord_sequence(self):
        """Record and detect multiple chords from terminal"""
        if not self.chord_library:
            print("\nNo chords in library! Please train some chords first.")
            return
        
        detected_chords = []
        
        print("\n" + "=" * 70)
        print("CHORD SEQUENCE DETECTOR")
        print("=" * 70)
        print("Instructions: Record chords one at a time.")
        print("Press 'd' to detect, 's' to save and exit, or 'q' to quit without saving.")
        print("=" * 70)
        
        while True:
            print(f"\nDetected so far: {len(detected_chords)} chords")
            if detected_chords:
                print(f"Chords: {', '.join(detected_chords)}")
            
            choice = input("\nOptions: (d)etect chord, (s)ave and exit, (q)uit: ").strip().lower()
            
            if choice == 'd':
                print("\nRecording chord...")
                audio = self.record_audio("Recording...")
                features = self.extract_features(audio)
                
                # Find best match
                best_match = None
                best_score = 0.0
                
                for chord_name, stored_features in self.chord_library.items():
                    score = self.compare_features(features, stored_features)
                    if score > best_score:
                        best_score = score
                        best_match = chord_name
                
                if best_score > 0.3 and best_match:
                    detected_chords.append(best_match)
                    print(f"✓ Detected: {best_match} ({best_score:.1%} confidence)")
                else:
                    print(f"✗ No confident match found (best score: {best_score:.1%})")
            
            elif choice == 's':
                if detected_chords:
                    # Remove duplicates while preserving order
                    unique_chords = []
                    seen = set()
                    for chord in detected_chords:
                        if chord not in seen:
                            unique_chords.append(chord)
                            seen.add(chord)
                    
                    # Save to JSON
                    sequence_data = {
                        "total_recorded": len(detected_chords),
                        "unique_chords": unique_chords,
                        "all_chords": detected_chords,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    filename = "chord_sequence.json"
                    with open(filename, 'w') as f:
                        json.dump(sequence_data, f, indent=2)
                    
                    print(f"\n✓ Saved {len(detected_chords)} chords ({len(unique_chords)} unique) to {filename}")
                    print(f"Unique chords: {', '.join(unique_chords)}")
                else:
                    print("\nNo chords detected to save.")
                break
            
            elif choice == 'q':
                print("\nExiting without saving...")
                break
            
            else:
                print("Invalid choice. Please try again.")
