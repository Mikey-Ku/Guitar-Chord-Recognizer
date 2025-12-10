"""
Main entry point for the Guitar Chord Recognizer
"""

from recognizer import ChordRecognizer


def main():
    """Main program interface"""
    print("=" * 60)
    print("Simple Guitar Chord Recognizer using Manual DFT")
    print("=" * 60)
    
    recognizer = ChordRecognizer(sample_rate=22050, duration=2.0)
    
    while True:
        print("\n" + "=" * 60)
        print("Options:")
        print("  1. Train chord database (record & store chord signatures)")
        print("  2. Detect chord sequence (interactive GUI)")
        print("  3. List trained chords")
        print("  4. Exit")
        print("=" * 60)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            chord_name = input("\nEnter chord name (e.g., C, G, Am, D7): ").strip()
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
            print("\nExiting... Goodbye!")
            break
        
        else:
            print("\nInvalid choice! Please enter 1-4.")


if __name__ == "__main__":
    main()
