"""
Song Recommender Module

Recommends songs based on learned chords and manages a song database.
"""

import json


class SongRecommender:
    """Recommends songs based on learned chords."""
    
    def __init__(self, songs_db_path="songs_database.json"):
        self.songs = []
        self.learned_chords = set()
        self.load_songs(songs_db_path)
    
    def load_songs(self, db_path):
        """Load songs from JSON database."""
        try:
            with open(db_path, 'r') as f:
                data = json.load(f)
                self.songs = data
        except Exception as e:
            print(f"Error loading songs database: {e}")
    
    def add_learned_chord(self, chord):
        """Add a chord to the set of learned chords."""
        self.learned_chords.add(chord)
    
    def get_playable_songs(self):
        """Get songs where ALL chords are learned."""
        playable = []
        
        for song in self.songs:
            song_chords = set(song['chords'])
            if song_chords.issubset(self.learned_chords):
                playable.append(song)
        
        return playable
    
    def get_almost_playable_songs(self):
        """Get songs missing less than 3 chords."""
        almost_playable = []
        
        for song in self.songs:
            song_chords = set(song['chords'])
            missing = song_chords - self.learned_chords
            
            if 0 < len(missing) < 3:
                almost_playable.append((song, missing))
        
        return almost_playable
