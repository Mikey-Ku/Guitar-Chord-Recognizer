"""
Main entry point for the Guitar Chord Recognizer
"""

from recognizer import ChordRecognizer
from song_recommender import SongRecommender
import json
import webbrowser
import os


def find_songs():
    """Find and display playable songs based on the last session."""
    recommender = SongRecommender()
    
    if not os.path.exists("chord_sequence.json"):
        print("\nNo chord sequence found. Please detect a sequence first.")
        return

    with open("chord_sequence.json", 'r') as f:
        sequence_data = json.load(f)
    
    learned_chords = set(sequence_data.get("unique_chords", []))

    if not learned_chords:
        print("\nNo unique chords found in the last session.")
        return

    for chord in learned_chords:
        recommender.add_learned_chord(chord)

    print("Song finder")
    
    playable = recommender.get_playable_songs()
    almost_playable = recommender.get_almost_playable_songs()
    
    print(f"\nChords from last recording session ({len(recommender.learned_chords)})")
    print(f"{', '.join(sorted(recommender.learned_chords))}")
    
    print(f"\nSONGS YOU CAN PLAY ({len(playable)})")
    
    if playable:
        for i, song in enumerate(playable, 1):
            print(f"\n  {i}. {song['song']} - {song['artist']}")
            print(f"Chords: {', '.join(song['chords'])}")
            
            choice = input(f"Open Youtube Link? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                try:
                    webbrowser.open(song['youtube'])
                    print(f"Opening youtube...")
                except Exception as e:
                    print(f"Could not open: {e}")
    else:
        print("None yet. Keep learning!")
    
        print(f"\nLearn less than 3 chords to play ({len(almost_playable)})")
    
    if almost_playable:
        for i, (song, missing) in enumerate(almost_playable, 1):
            missing_list = sorted(missing)
            print(f"\n  {i}. {song['song']} - {song['artist']}")
            print(f"Chords needed: {', '.join(missing_list)}")
            
            choice = input(f"Open youtube? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                try:
                    webbrowser.open(song['youtube'])
                    print(f"Opening youtube...")
                except Exception as e:
                    print(f"Could not open: {e}")
    else:
        print("None. You can play all songs!")


def main():
    """Main program interface"""
    print("Simple Guitar Chord Recognizer using Manual DFT")
    
    recognizer = ChordRecognizer(sample_rate=22050, duration=2.0)
    
    while True:
        print("\n" + "=" * 60)
        print("Options:")
        print("1. Train chord database (record & store chord signatures)")
        print("2. Detect chord sequence (interactive GUI)")
        print("3. List trained chords")
        print("4. Find songs from last session")
        print("5. Change microphone/input device")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            chord_name = input("\nEnter chord name: ").strip()
            if chord_name:
                recognizer.train_chord(chord_name)
            else:
                print("Invalid chord name!")
        
        elif choice == "2":
            print("\nOpening chord sequence detector window...")
            print("Click 'Record Chord' button to detect each chord you play.")
            print("The window will show all detected chords.")
            recognizer.detect_chord_sequence()
        
        elif choice == "3":
            recognizer.list_chords()

        elif choice == "4":
            find_songs()
        
        elif choice == "5":
            print("\nExiting... Goodbye!")
            break
        
        else:
            print("\nInvalid choice! Please enter 1-5.")


if __name__ == "__main__":
    main()
