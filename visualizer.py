import os
import json
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys

CHORD_RECORDINGS_PATH = "chord_recordings"
ANALYSIS_OUTPUT_PATH = "analysis"

class WaveVisualizer:
    """Interactive visualizer for chord waveforms and frequency spectra."""
    
    def __init__(self):
        self.chords = []
        self.current_chord_idx = 0
        self.samples = {}
        self.current_sample = 0
        self.load_chords()
        
        if not self.chords:
            print("No chord recordings found! Make sure to run chord_collector.py first.")
            sys.exit(1)
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("Guitar Chord Wave Analyzer", fontsize=16, fontweight='bold')
        
        # Waveform plot
        self.ax_wave = plt.subplot(2, 1, 1)
        self.ax_wave.set_title("Time Domain Waveform")
        self.ax_wave.set_xlabel("Time (seconds)")
        self.ax_wave.set_ylabel("Amplitude")
        
        # Frequency spectrum plot
        self.ax_freq = plt.subplot(2, 1, 2)
        self.ax_freq.set_title("Frequency Domain (DFT)")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Magnitude")
        
        # Navigation buttons
        ax_prev_chord = plt.axes([0.2, 0.05, 0.08, 0.04])
        ax_next_chord = plt.axes([0.3, 0.05, 0.08, 0.04])
        ax_prev_sample = plt.axes([0.5, 0.05, 0.08, 0.04])
        ax_next_sample = plt.axes([0.6, 0.05, 0.08, 0.04])
        
        self.btn_prev_chord = Button(ax_prev_chord, '< Chord')
        self.btn_next_chord = Button(ax_next_chord, 'Chord >')
        self.btn_prev_sample = Button(ax_prev_sample, '< Sample')
        self.btn_next_sample = Button(ax_next_sample, 'Sample >')
        
        self.btn_prev_chord.on_clicked(self.prev_chord)
        self.btn_next_chord.on_clicked(self.next_chord)
        self.btn_prev_sample.on_clicked(self.prev_sample)
        self.btn_next_sample.on_clicked(self.next_sample)
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.01, '', ha='center', fontsize=10)
        
        # Draw initial plot
        self.update_plot()
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.96])
        plt.show()
    
    def load_chords(self):
        """Load available chord folders."""
        if os.path.exists(CHORD_RECORDINGS_PATH):
            self.chords = sorted([d for d in os.listdir(CHORD_RECORDINGS_PATH) 
                                 if os.path.isdir(os.path.join(CHORD_RECORDINGS_PATH, d))])
        
        # Load samples for each chord
        for chord in self.chords:
            chord_folder = os.path.join(CHORD_RECORDINGS_PATH, chord)
            wav_files = sorted([f for f in os.listdir(chord_folder) if f.endswith('.wav')])
            self.samples[chord] = wav_files
    
    def get_current_chord(self):
        """Get current chord name."""
        return self.chords[self.current_chord_idx] if self.chords else None
    
    def get_current_sample_path(self):
        """Get full path to current sample."""
        chord = self.get_current_chord()
        if not chord or not self.samples[chord]:
            return None
        
        sample_file = self.samples[chord][self.current_sample]
        return os.path.join(CHORD_RECORDINGS_PATH, chord, sample_file)
    
    def get_analysis_json(self):
        """Get JSON analysis for current sample if it exists."""
        chord = self.get_current_chord()
        if not chord or not self.samples[chord]:
            return None
        
        sample_file = self.samples[chord][self.current_sample]
        json_file = sample_file.replace('.wav', '.json')
        json_path = os.path.join(ANALYSIS_OUTPUT_PATH, chord, json_file)
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return None
    
    def update_plot(self):
        """Update both waveform and frequency plots."""
        sample_path = self.get_current_sample_path()
        chord = self.get_current_chord()
        
        if not sample_path or not os.path.exists(sample_path):
            self.status_text.set_text("No sample found")
            return
        
        # Read audio file
        sample_rate, audio_data = read(sample_path)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Time axis
        time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        # Clear previous plots
        self.ax_wave.clear()
        self.ax_freq.clear()
        
        # Plot waveform
        self.ax_wave.plot(time, audio_data, linewidth=0.5, color='steelblue')
        self.ax_wave.set_title(f"Time Domain Waveform - {chord}")
        self.ax_wave.set_xlabel("Time (seconds)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.3)
        
        # Get analysis data
        analysis = self.get_analysis_json()
        
        if analysis:
            # Plot frequency spectrum from saved analysis
            frequencies = np.array(analysis['frequencies_hz'])
            magnitudes = np.array(analysis['dft_magnitudes'])
            
            # Focus on lower frequencies for better visualization
            freq_limit = 5000  # Hz
            mask = frequencies < freq_limit
            
            self.ax_freq.plot(frequencies[mask], magnitudes[mask], linewidth=1, color='darkred')
            self.ax_freq.set_title(f"Frequency Domain (DFT) - Peak: {analysis['peak_frequency_hz']:.2f} Hz")
            self.ax_freq.set_xlabel("Frequency (Hz)")
            self.ax_freq.set_ylabel("Magnitude")
            self.ax_freq.grid(True, alpha=0.3)
            
            # Add peak frequency marker
            peak_freq = analysis['peak_frequency_hz']
            peak_mag = analysis['peak_magnitude']
            self.ax_freq.plot(peak_freq, peak_mag, 'go', markersize=8, label=f'Peak: {peak_freq:.2f} Hz')
            self.ax_freq.legend()
        else:
            # Compute DFT on the fly if no analysis exists
            from dft import dft
            dft_magnitudes = dft(audio_data)
            frequencies = np.abs(np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(dft_magnitudes)])
            
            freq_limit = 5000
            mask = frequencies < freq_limit
            
            self.ax_freq.plot(frequencies[mask], dft_magnitudes[mask], linewidth=1, color='darkred')
            self.ax_freq.set_title("Frequency Domain (DFT)")
            self.ax_freq.set_xlabel("Frequency (Hz)")
            self.ax_freq.set_ylabel("Magnitude")
            self.ax_freq.grid(True, alpha=0.3)
        
        # Update status
        sample_name = self.samples[chord][self.current_sample]
        self.status_text.set_text(
            f"Chord: {chord} | Sample: {self.current_sample + 1}/{len(self.samples[chord])} ({sample_name})"
        )
        
        self.fig.canvas.draw_idle()
    
    def prev_chord(self, event):
        """Navigate to previous chord."""
        self.current_chord_idx = (self.current_chord_idx - 1) % len(self.chords)
        self.current_sample = 0
        self.update_plot()
    
    def next_chord(self, event):
        """Navigate to next chord."""
        self.current_chord_idx = (self.current_chord_idx + 1) % len(self.chords)
        self.current_sample = 0
        self.update_plot()
    
    def prev_sample(self, event):
        """Navigate to previous sample."""
        chord = self.get_current_chord()
        if chord and len(self.samples[chord]) > 0:
            self.current_sample = (self.current_sample - 1) % len(self.samples[chord])
            self.update_plot()
    
    def next_sample(self, event):
        """Navigate to next sample."""
        chord = self.get_current_chord()
        if chord and len(self.samples[chord]) > 0:
            self.current_sample = (self.current_sample + 1) % len(self.samples[chord])
            self.update_plot()


if __name__ == "__main__":
    visualizer = WaveVisualizer()
