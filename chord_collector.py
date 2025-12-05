import os
from audio_recorder import audio_recorder

major_chords = [
    "A", "A#", "Bb",
    "B", "Cb",
    "C", "C#", "Db",
    "D", "D#", "Eb",
    "E", "Fb",
    "F", "F#", "Gb",
    "G", "G#", "Ab"
]

SAMPLES_PER_CHORD = 5
BASE_FOLDER = "chord_recordings"

def collect_chord_samples():
    """
    Collects samples for each chord and organizes them into unique folders.
    Each chord gets its own folder with 5 recorded samples.
    """
    # Create base folder if it doesn't exist
    if not os.path.exists(BASE_FOLDER):
        os.makedirs(BASE_FOLDER)
    
    for chord in major_chords:
        chord_folder = os.path.join(BASE_FOLDER, chord)
        if not os.path.exists(chord_folder):
            os.makedirs(chord_folder)
        
        # Record 5 samples for each chord
        for sample_num in range(1, SAMPLES_PER_CHORD + 1):
            print(f"\nRecording {chord} - Sample {sample_num}/{SAMPLES_PER_CHORD}")
            filename = f"{chord}_sample_{sample_num}.wav"
            audio_recorder(chord, folder=chord_folder, filename=filename)
            print(f"Saved: {os.path.join(chord_folder, filename)}")

if __name__ == "__main__":
    collect_chord_samples()