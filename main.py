import argparse
import random
import pygame
import numpy as np
import time
from collections import deque
import threading
import hashlib
from tqdm import tqdm
from pathlib import Path
from thefuzz import process
from pynput import keyboard
from music_embedder import audio_embed

class MusicPlayer:
    def __init__(self, music_directory):
        self.music_directory = Path(music_directory)
        self.cache_directory = Path(__file__).parent / "cache"
        
        self.initialize_embeddings()
        
        self.current_embedding = None
        self.current_track_index = 0
        self.is_playing = False
        self.history = deque(maxlen=len(self.playlist_paths) - 1)
        self.recently_played = deque(maxlen=len(self.playlist_paths) - 1) # used to prevent duplicates for next nearest neighbors
        self.should_exit = False
        pygame.mixer.init()
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        pygame.init()

    def initialize_embeddings(self):
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        self.playlist_paths = []
        self.music_embeddings = []

        current_files = list(self.music_directory.rglob('*.mp3'))

        print("Initializing embeddings...")
        for file in tqdm(current_files, desc="Processing files", unit="file"):
            file_hash = hashlib.md5(file.read_bytes()).hexdigest()
            cache_file = self.cache_directory / f"{file_hash}.npz"

            if cache_file.exists():
                # Load from cache
                with np.load(cache_file) as data:
                    embedding = data['embedding']
            else:
                # Create new embedding
                embedding = audio_embed(str(file))
                np.savez_compressed(cache_file, embedding=embedding, file_path=str(file))

            self.playlist_paths.append(file)
            self.music_embeddings.append(embedding)

        self.music_embeddings = np.array(self.music_embeddings)
        print(f"Found and processed {len(self.playlist_paths)} songs.")

    def toggle_play_pause(self):
        if not self.is_playing:
            self.is_playing = True
            if not pygame.mixer.music.get_busy():
                self.play_current_track()
            else:
                pygame.mixer.music.unpause()
                print("Music unpaused.")
        else:
            pygame.mixer.music.pause()
            self.is_playing = False
            print("Music paused.")

    def play_current_track(self):
        self.is_playing = True
        self.current_embedding = self.music_embeddings[self.current_track_index]
        pygame.mixer.music.load(self.playlist_paths[self.current_track_index])
        pygame.mixer.music.play()
        self.history.append(self.current_track_index)
        self.recently_played.append(self.current_track_index)
        print(f"Playing: {Path(self.playlist_paths[self.current_track_index]).name}")

    def stop_music(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        print("Music stopped.")

    def next_track(self, similar=False):
        if similar:
            self.current_track_index = self.find_nearest_embedding()
            self.current_embedding = self.music_embeddings[self.current_track_index]
        else:
            self.current_track_index = random.randint(0, len(self.playlist_paths) - 1)
            # Reset recently played tracks on random shuffle
            self.recently_played.clear()
        
        if self.is_playing:
            self.play_current_track()
        else:
            print(f"Next track selected: {self.playlist_paths[self.current_track_index].name}")

    def find_nearest_embedding(self):
        distances = np.linalg.norm(self.music_embeddings - self.current_embedding, axis=1)
        # Prevent looping of similar songs
        for idx in self.recently_played:
            distances[idx] = np.inf
        return np.argmin(distances)

    def previous_track(self):
        if len(self.history) > 1:
            self.history.pop()
            self.current_track_index = self.history.pop()
        else:
            self.current_track_index = (self.current_track_index - 1) % len(self.playlist_paths)
        
        if self.is_playing:
            self.play_current_track()
        else:
            print(f"Previous track selected: {self.playlist_paths[self.current_track_index].name}")

    def shuffle_playlist(self):
        combined = list(zip(self.playlist_paths, self.music_embeddings))
        random.shuffle(combined)
        self.playlist_paths, self.music_embeddings = zip(*combined)
        self.playlist_paths = list(self.playlist_paths)
        self.music_embeddings = np.array(self.music_embeddings)
        self.recently_played.clear()
        print("Playlist shuffled.")

    def fuzzy_search(self, query):
        song_names = [p.stem for p in self.playlist_paths]
        results = process.extract(query, song_names, limit=10)
        return results
    
    def play_song_by_index(self, index):
        if 0 <= index < len(self.playlist_paths):
            self.current_track_index = index
            self.play_current_track()
        else:
            print("Invalid song index.")

    def check_music_end(self):
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:
                self.next_track(similar=True)

    def run(self):
        while not self.should_exit:
            if self.is_playing:
                self.check_music_end()
            time.sleep(0.1)

def print_menu():
    print("\nVibeShuffle Commands:")
    print("p - Play/Pause music")
    print("s - Stop music")
    print("n - Play next track (random)")
    print("m - Play next track (similar)")
    print("b - Play previous track")
    print("r - Shuffle playlist")
    print("f - Fuzzy search for a song")
    print("q - Exit the player")

def handle_user_input(player):
    print_menu()
    while not player.should_exit:
        command = input("Enter command: ").lower().strip()
        if command == "p":
            player.toggle_play_pause()
        elif command == "s":
            player.stop_music()
        elif command == "n":
            player.next_track(similar=False)
        elif command == "m":
            player.next_track(similar=True)
        elif command == "b":
            player.previous_track()
        elif command == "r":
            player.shuffle_playlist()
        elif command == "f":
            query = input("Enter search query: ")
            results = player.fuzzy_search(query)
            print("\nTop 5 matches:")
            for i, (song, score) in enumerate(results):
                print(f"{i+1}. {song} (Score: {score})")
            choice = input("Enter the number of the song you want to play (or press Enter to cancel): ")
            if choice.isdigit() and 1 <= int(choice) <= 10:
                index = [p.name for p in player.playlist_paths].index(f"{results[int(choice)-1][0]}.mp3")
                player.play_song_by_index(index)
        elif command == "q":
            print("Exiting Music Player.")
            player.should_exit = True
        else:
            print("Invalid command. Try again.")



def main():
    parser = argparse.ArgumentParser(description="VibeShuffle Music Player")
    parser.add_argument("music_directory", type=str, help="Path to the directory containing music files (mp3)")
    args = parser.parse_args()

    music_directory = args.music_directory
    player = MusicPlayer(music_directory)
    player.shuffle_playlist()

    input_thread = threading.Thread(target=handle_user_input, args=(player,))
    input_thread.start()

    def on_press(key):
        if key == keyboard.Key.f5:
            player.next_track(similar=False)
        elif key == keyboard.Key.f6:
            player.next_track(similar=True)
        elif key == keyboard.Key.media_play_pause:
            player.toggle_play_pause()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    player.run()

if __name__ == "__main__":
    main()
