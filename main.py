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
from music_embedder import audio_embed

class MusicPlayer:
    def __init__(self, music_directory):
        self.music_directory = Path(music_directory)
        self.cache_directory = Path(__file__).parent / "cache"
        
        self.initialize_embeddings()
        
        self.current_embedding = None
        self.current_track_index = 0
        self.is_playing = False
        self.recently_played = deque(maxlen=len(self.playlist) - 1)
        self.should_exit = False
        pygame.mixer.init()
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        pygame.init()

    def initialize_embeddings(self):
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        self.playlist = []
        self.music_embeddings = []

        current_files = list(self.music_directory.glob('*.mp3'))

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

            self.playlist.append(str(file))
            self.music_embeddings.append(embedding)

        self.music_embeddings = np.array(self.music_embeddings)
        print(f"Found and processed {len(self.playlist)} songs.")

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
        self.current_embedding = self.music_embeddings[self.current_track_index]
        pygame.mixer.music.load(self.playlist[self.current_track_index])
        pygame.mixer.music.play()
        self.recently_played.append(self.current_track_index)
        print(f"Playing: {Path(self.playlist[self.current_track_index]).name}")

    def stop_music(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        print("Music stopped.")

    def next_track(self, similar=False):
        if similar:
            self.current_track_index = self.find_nearest_embedding()
        else:
            self.current_track_index = random.randint(0, len(self.playlist) - 1)
            # Reset recently played tracks on random shuffle
            self.recently_played.clear()
        
        if self.is_playing:
            self.play_current_track()
        else:
            print(f"Next track selected: {Path(self.playlist[self.current_track_index]).name}")

    def find_nearest_embedding(self):
        distances = np.linalg.norm(self.music_embeddings - self.current_embedding, axis=1)
        # Prevent looping of similar songs
        for idx in self.recently_played:
            distances[idx] = np.inf
        return np.argmin(distances)

    def previous_track(self):
        if len(self.recently_played) > 1:
            self.recently_played.pop()
            self.current_track_index = self.recently_played.pop()
        else:
            self.current_track_index = (self.current_track_index - 1) % len(self.playlist)
        
        if self.is_playing:
            self.play_current_track()
        else:
            print(f"Previous track selected: {Path(self.playlist[self.current_track_index]).name}")

    def shuffle_playlist(self):
        combined = list(zip(self.playlist, self.music_embeddings))
        random.shuffle(combined)
        self.playlist, self.music_embeddings = zip(*combined)
        self.playlist = list(self.playlist)
        self.music_embeddings = np.array(self.music_embeddings)
        self.recently_played.clear()
        print("Playlist shuffled.")

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

    player.run()
    input_thread.join()

if __name__ == "__main__":
    main()
