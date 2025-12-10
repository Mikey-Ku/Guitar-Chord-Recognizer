"""
Chord recognition system using manual DFT
"""

import time
import sounddevice as sd
import json
import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
from dft import ManualDFT


class ChordRecognizer:
    """Main chord recognizer class"""
    
    def __init__(self, sample_rate=44100, duration=2.0):
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
    
    def train_chord(self, chord_name):
        """Record and store a chord in the library"""
        audio = self.record_audio(f"Recording chord: {chord_name}")
        features = self.extract_features(audio)
        
        self.chord_library[chord_name] = features
        print(f"\nChord '{chord_name}' features extracted:")
        print("Top frequencies (Hz) and magnitudes:")
        for freq, mag in features[:5]:
            print(f"  {freq:.1f} Hz: {mag:.2f}")
        
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
        """Interactive GUI for recording and detecting multiple chords"""
        if not self.chord_library:
            print("\nNo chords in library! Please train some chords first.")
            return
        
        detected_chords = []
        
        # Create GUI window
        window = tk.Tk()
        window.title("Chord Sequence Detector")
        window.geometry("600x500")
        
        # Results display
        results_label = tk.Label(window, text="Detected Chords:", font=("Arial", 14, "bold"))
        results_label.pack(pady=10)
        
        results_text = scrolledtext.ScrolledText(window, height=15, width=60, font=("Arial", 12))
        results_text.pack(pady=10)
        
        status_label = tk.Label(window, text="Ready to record", font=("Arial", 11))
        status_label.pack(pady=5)
        
        def record_and_detect():
            """Record a chord and detect it"""
            status_label.config(text="Recording...", fg="red")
            window.update()
            
            try:
                # Record audio
                audio = self.record_audio_silent()
                
                # Extract features
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
                    results_text.insert(tk.END, f"{len(detected_chords)}. {best_match} ({best_score:.1%} confidence)\n")
                    results_text.see(tk.END)
                    status_label.config(text=f"Detected: {best_match} - Ready for next chord", fg="green")
                else:
                    status_label.config(text="No match found - Try again", fg="orange")
                    
            except Exception as e:
                status_label.config(text=f"Error: {str(e)}", fg="red")
            
            window.update()
        
        def save_and_close():
            """Save detected chords and close window"""
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
                
                messagebox.showinfo("Saved!", 
                    f"Saved {len(detected_chords)} chords ({len(unique_chords)} unique)\n"
                    f"File: {filename}\n"
                    f"Unique chords: {', '.join(unique_chords)}")
            
            window.destroy()
        
        # Buttons
        button_frame = tk.Frame(window)
        button_frame.pack(pady=10)
        
        record_btn = tk.Button(button_frame, text="Record Chord", 
                               command=record_and_detect,
                               font=("Arial", 12, "bold"),
                               bg="#4CAF50", fg="white",
                               width=15, height=2)
        record_btn.pack(side=tk.LEFT, padx=10)
        
        save_btn = tk.Button(button_frame, text="Save & Close",
                            command=save_and_close,
                            font=("Arial", 12, "bold"),
                            bg="#2196F3", fg="white",
                            width=15, height=2)
        save_btn.pack(side=tk.LEFT, padx=10)
        
        # Instructions
        instructions = tk.Label(window, 
                               text="Click 'Record Chord' to detect each chord.\n"
                                    "Record as many as you want, then click 'Save & Close'.",
                               font=("Arial", 10), fg="gray")
        instructions.pack(pady=10)
        
        window.mainloop()
    
    def record_audio_silent(self):
        """Record audio without printing messages (for GUI)"""
        time.sleep(0.5)  # Short pause
        
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        return audio.flatten()
