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
        self.input_device = None

        self.load_library()
        
    def _check_audio_devices(self):
        """Check and display available audio input devices"""
        try:
            devices = sd.query_devices()
            print("\n" + "="*60)
            print("Detected Audio Devices:")
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device))
                    marker = "[Default]" if i == sd.default.device[0] else ""
                    print(f"  {i}: {device['name']} {marker}")
            
            if not input_devices:
                print("No input devices found")
            print("="*60 + "\n")
        except Exception as e:
            print(f"Warning: Could not get audio devices: {e}")
    
    def record_audio(self, message="Recording...", check_volume=True):
        """Record audio from microphone with volume checking"""
        print("\nPausing for 1 second before recording...")
        time.sleep(1)
    
        print(f"\n{message}")
        print(f"Recording for {self.duration} seconds...")
        
        if self.input_device is not None:
            print(f"Using device: {sd.query_devices(self.input_device)['name']}")
        else:
            print(f"Using default input device")
        
        try:
            audio = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.input_device
            )
            sd.wait()
            print("Recording complete!")

            audio = audio.flatten()
            
            if check_volume:
                rms = np.sqrt(np.mean(audio**2))
                peak = np.max(np.abs(audio))
                print(f"\nRecording levels: RMS={rms:.6f}, Peak={peak:.6f}")
                
                if peak < 0.001:
                    print("\n WARNING: Recording is almost silent!")
                    print("Possible issues:")
                    print("- Microphone is muted or disabled")
                    print("- Wrong microphone selected")
                    print("- Need to grant microphone permissions")
                    print("- Guitar/sound source is too quiet or far away")
                    
                    retry = input("\nTry recording again? (y/n): ").strip().lower()
                    if retry in ['y', 'yes']:
                        return self.record_audio(message, check_volume=True)
                elif rms < 0.01:
                    print("Note: Recording is quiet - get closer to the microphone")
            
            return audio
            
        except Exception as e:
            print(f"\nRecording error: {e}")
            print("Check your microphone permissions in System Settings")
            return None
    
    def extract_features(self, audio_signal):
        """
        Extract frequency features from audio using manual DFT
        Returns the top frequency peaks and their magnitudes
        """
        target_duration = 0.5  # seconds
        target_samples = int(target_duration * self.sample_rate)
        
        downsample_factor = max(1, len(audio_signal) // target_samples)
        
        audio_downsampled = audio_signal[::downsample_factor][:target_samples]
        
        print(f"Processing {len(audio_downsampled)} samples from {len(audio_signal)} (every {downsample_factor}th sample)...")
        
        frequencies = self.dft.dft(audio_downsampled)
        magnitudes = self.dft.get_magnitude_spectrum(frequencies)
        
        freq_bins = [(i * self.sample_rate / len(audio_downsampled), magnitudes[i]) 
                     for i in range(len(magnitudes))]
        
        freq_bins.sort(key=lambda x: x[1], reverse=True)
        
        top_peaks = freq_bins[:10]
        
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
        """Plot two DFT magnitude spectra: downsampled spread vs continuous 0.5s"""
        target_duration = 0.5
        target_samples = int(target_duration * self.sample_rate)

        downsample_factor = max(1, len(audio_signal) // target_samples)
        spread_signal = audio_signal[::downsample_factor][:target_samples]

        contiguous_signal = audio_signal[:target_samples]

        freq_spread, mag_spread = self._compute_spectrum(spread_signal)
        freq_contig, mag_contig = self._compute_spectrum(contiguous_signal)

        max_mag_spread = max(mag_spread) if mag_spread else 0
        max_mag_contig = max(mag_contig) if mag_contig else 0
        
        if max_mag_spread < 1.0 and max_mag_contig < 1.0:
            print("\n WARNING: DFT magnitude is very low - signal may be too quiet")

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(freq_spread, mag_spread, color='tab:blue', linewidth=1.5)
        plt.title(f"{chord_name} - Downsampled (spread)\nMax magnitude: {max_mag_spread:.1f}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("DFT Magnitude")
        plt.xlim(0, 2000)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(freq_contig, mag_contig, color='tab:orange', linewidth=1.5)
        plt.title(f"{chord_name} - Contiguous (first 0.5s)\nMax magnitude: {max_mag_contig:.1f}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("DFT Magnitude")
        plt.xlim(0, 2000)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        os.makedirs("dft_plots", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("dft_plots", f"{chord_name}_{ts}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\n Saved DFT plots to: {out_path}")
        print(f"Open this file to view the frequency analysis")

    def train_chord(self, chord_name):
        """Record and store a chord in the library"""
        audio = self.record_audio(f"Recording chord: {chord_name}")
        
        if audio is None:
            print("\nRecording failed - chord not saved")
            return
        
        features = self.extract_features(audio)
        
        if not features:
            print("\nNo features extracted - recording may be silent")
            return

        self.chord_library[chord_name] = features
        print(f"\nChord '{chord_name}' features extracted:")
        print("Top frequencies (Hz) and magnitudes:")
        for freq, mag in features[:5]:
            print(f"  {freq:.1f} Hz: {mag:.2f}")

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
        tolerance = 50
        
        for freq1, mag1 in features1[:6]:
            best_match = 0.0
            for freq2, mag2 in features2[:6]:
                if abs(freq1 - freq2) < tolerance:
                    freq_similarity = 1.0 - (abs(freq1 - freq2) / tolerance)
                    mag_similarity = min(mag1, mag2) / max(mag1, mag2) if max(mag1, mag2) > 0 else 0
                    match = freq_similarity * mag_similarity
                    best_match = max(best_match, match)
            score += best_match
        
        return score / 6.0
    
    def recognize_chord(self):
        """Record live audio and identify the chord"""
        if not self.chord_library:
            print("\nNo chords in library! Please train some chords first.")
            return None
        
        audio = self.record_audio("Recording for chord recognition...")
        
        if audio is None:
            print("\nRecording failed")
            return None
        
        features = self.extract_features(audio)
        
        print("\nDetected frequencies:")
        for freq, mag in features[:5]:
            print(f"  {freq:.1f} Hz: {mag:.2f}")
        
        best_match = None
        best_score = 0.0
        
        print("\nMatching against library...")
        for chord_name, stored_features in self.chord_library.items():
            score = self.compare_features(features, stored_features)
            print(f"  {chord_name}: {score:.2%} match")
            
            if score > best_score:
                best_score = score
                best_match = chord_name
        
        if best_score > 0.3:
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
                
                if audio is None:
                    print("Recording failed - try again")
                    continue
                
                features = self.extract_features(audio)
                
                best_match = None
                best_score = 0.0
                
                for chord_name, stored_features in self.chord_library.items():
                    score = self.compare_features(features, stored_features)
                    if score > best_score:
                        best_score = score
                        best_match = chord_name
                
                if best_score > 0.3 and best_match:
                    detected_chords.append(best_match)
                    print(f"Detected: {best_match} ({best_score:.1%} confidence)")
                else:
                    print(f"No confident match found (best score: {best_score:.1%})")
            
            elif choice == 's':
                if detected_chords:
                    unique_chords = []
                    seen = set()
                    for chord in detected_chords:
                        if chord not in seen:
                            unique_chords.append(chord)
                            seen.add(chord)
                    
                    sequence_data = {
                        "total_recorded": len(detected_chords),
                        "unique_chords": unique_chords,
                        "all_chords": detected_chords,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    filename = "chord_sequence.json"
                    with open(filename, 'w') as f:
                        json.dump(sequence_data, f, indent=2)
                    
                    print(f"\nSaved {len(detected_chords)} chords ({len(unique_chords)} unique) to {filename}")
                    print(f"Unique chords: {', '.join(unique_chords)}")
                else:
                    print("\nNo chords detected to save.")
                break
            
            elif choice == 'q':
                print("\nExiting without saving...")
                break
            
            else:
                print("Invalid choice. Please try again.")
