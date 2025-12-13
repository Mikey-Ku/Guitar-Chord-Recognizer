# Simple Guitar Chord Recognizer using Manual DFT

A simple guitar chord recognition system that uses a **manually implemented DFT** (no FFT libraries) to identify chords by matching frequency patterns. This tool also includes a song recommender to help you find songs to play with the chords you've learned.

## Features

- **Manual DFT Implementation**: The DFT is computed from scratch using basic trigonometric functions.
- **Chord Training**: Record and store multiple guitar chords to build a recognition database.
- **Chord Sequence Detection**: Detect a sequence of chords played one after another.
- **Song Recommender**: Get recommendations for songs you can play based on the chords you know.
- **Persistent Storage**: 
    - Trained chords are saved to `chord_library.json`.
    - Detected chord sequences are saved to `chord_sequence.json`.
- **Informative Plots**: Generates and saves DFT plots for analysis during training.

## Project Structure

```
├── main.py                 # Entry point with menu interface
├── recognizer.py           # Chord recognition logic
├── dft.py                  # Manual DFT implementation
├── song_recommender.py     # Song recommendation logic
├── requirements.txt        # Python dependencies
├── chord_library.json      # Stores trained chord signatures
├── songs_database.json     # Contains songs for the recommender
├── chord_sequence.json     # Stores the last detected chord sequence
├── dft_plots/              # Directory for DFT magnitude spectrum plots
└── README.md               # This file
```

## How It Works

### Phase 1: Training (Build Chord Database)
Record individual guitar chords to build a signature database. The system:
   - Records 2 seconds of audio.
   - Downsamples the audio to speed up DFT calculation.
   - Performs a manual DFT to extract frequency components.
   - Identifies the top frequency peaks and their magnitudes (in the 80-1500 Hz range).
   - Stores this "signature" for each chord in `chord_library.json`.

### Phase 2: Chord Sequence Detection
Detect a sequence of chords:
   - The system records chords one at a time.
   - It compares the recorded chord's frequency signature to the trained chords in the library.
   - The recognized chords are displayed and saved to `chord_sequence.json`.

### Phase 3: Song Recommendation
After a chord sequence is detected, the system can:
- Identify the unique chords from the last session.
- Search the `songs_database.json` for songs that can be played with the learned chords.
- Suggest songs that are "almost playable" (requiring the user to learn only one or two new chords).

## Installation

1.  Clone the repository.
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the program from your terminal:

```bash
python main.py
```

### Workflow:

1.  **Option 1 - Train chord database**: Record individual chords (e.g., C, G, Am, D7) to build your signature library. It's recommended to train at least 3-5 different chords.
2.  **Option 2 - Detect chord sequence**: Record a sequence of chords. The system will identify each chord and save the sequence.
3.  **Option 3 - List trained chords**: View all chords currently in your database.
4.  **Option 4 - Find songs from last session**: Get song recommendations based on the last detected chord sequence.
5.  **Option 5 - Exit**: Close the application.

## Tips for Best Results

- Play chords clearly and let them ring for the full 2 seconds.
- Maintain a consistent distance from the microphone during training and detection.
- Train in a quiet environment to minimize background noise.
- Use the same guitar for training and testing if possible.

## Output Files

- **`chord_library.json`**: Contains all trained chord signatures (frequencies and magnitudes).
- **`chord_sequence.json`**: Contains the last detected chord sequence, including all chords detected and a list of unique chords.
- **`dft_plots/*.png`**: PNG images of the DFT magnitude spectrum generated during training.

## Technical Details

- **Sample Rate**: 22,050 Hz
- **Recording Duration**: 2 seconds
- **Frequency Range**: 80-1500 Hz (for chord recognition)
- **DFT**: Manually computed using `cos`/`sin` functions.
- **Matching**: Uses frequency tolerance with magnitude weighting.
- **Confidence Threshold**: A configurable threshold (currently 30%) for valid chord recognition.