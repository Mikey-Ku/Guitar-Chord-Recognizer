# Simple Guitar Chord Recognizer using Manual DFT

A simple guitar chord recognition system that uses a **manually implemented DFT** (no FFT libraries) to identify chords by matching frequency patterns.

## Features

- **Manual DFT Implementation**: The DFT is computed from scratch using basic trigonometric functions
- **Chord Training Database**: Record and store multiple guitar chords for recognition
- **Interactive Chord Sequence Detection**: GUI-based interface to detect multiple chords in succession
- **Smart Downsampling**: Processes 0.5 seconds of data spread across the full recording for 16x faster DFT
- **Persistent Storage**: Trained chords saved to `chord_library.json`, detected sequences saved to `chord_sequence.json`
- **Duplicate Removal**: Automatically filters unique chords in sequence detection

## Project Structure

```
├── main.py          # Entry point with menu interface
├── recognizer.py    # Chord recognition logic and GUI
├── dft.py          # Manual DFT implementation
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## How It Works

### Phase 1: Training (Build Chord Database)
Record individual guitar chords to build a signature database. The system:
   - Records 2 seconds of audio
   - Downsamples to 0.5 seconds of data (every Nth sample across full recording)
   - Performs manual DFT to extract frequency components
   - Identifies the top frequency peaks and their magnitudes (80-1500 Hz range)
   - Stores this "signature" for each chord in `chord_library.json`

### Phase 2: Chord Sequence Detection (Interactive GUI)
Opens a popup window where you can:
   - Click "Record Chord" to detect each chord you play
   - See real-time results as chords are identified
   - Record as many chords as you want in succession
   - Click "Save & Close" to save the sequence to `chord_sequence.json`
   - Automatically removes duplicates and shows unique chords

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the program:

```bash
python main.py
```

### Workflow:

1. **Option 1 - Train Chord Database**: Record individual chords (C, G, Am, D7, etc.) to build your signature library. Do this until you're satisfied with your database.

2. **Option 2 - Detect Chord Sequence**: Opens an interactive GUI window
   - Click "Record Chord" button
   - Play a chord and let it ring for 2 seconds
   - System identifies and displays the chord
   - Repeat for as many chords as you want
   - Click "Save & Close" when done
   - Results saved to `chord_sequence.json` with unique chords (no duplicates)

3. **Option 3 - List Trained Chords**: View all chords currently in your database

4. **Option 4 - Exit**: Close the application

## Tips for Best Results

- Play chords clearly and let them ring for the full 2 seconds
- Keep consistent distance from microphone during training and detection
- Train in a quiet environment to minimize background noise
- Use the same guitar for training and testing if possible
- Build a library of at least 3-5 different chords before using sequence detection
- In the GUI, wait for the status to say "Ready" before clicking "Record Chord" again

## Output Files

- **`chord_library.json`**: Contains all trained chord signatures (frequencies and magnitudes)
- **`chord_sequence.json`**: Contains detected chord sequences with:
  - `unique_chords`: List of unique chords (no duplicates)
  - `all_chords`: Full sequence of detected chords
  - `total_recorded`: Total number of chords detected
  - `timestamp`: When the sequence was recorded

## Technical Details

- **Sample Rate**: 22,050 Hz (sufficient for guitar frequencies)
- **Duration**: 2 seconds per recording
- **Downsampling**: Reduces to 0.5 seconds worth of data (every 4th sample) for ~16x speedup
- **Frequency Range**: 80-1500 Hz (typical guitar range)
- **DFT**: Manually computed using cos/sin functions, no FFT library
- **Matching**: Uses frequency tolerance (±50 Hz) with magnitude weighting
- **Confidence Threshold**: 30% minimum for valid chord recognition
- **GUI Framework**: tkinter (included with Python)
