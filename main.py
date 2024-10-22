import argparse
import random
import time
from pathlib import Path
from collections import deque
import threading
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import numpy as np
from tqdm import tqdm
from rapidfuzz import process, fuzz
from pynput import keyboard


class MusicPlayer:
    def __init__(self, music_directory):
        self.music_directory = Path(music_directory)
        self.cache_directory = Path(__file__).parent / "cache"

        self.initialize_embeddings()

        self.current_embedding = None
        self.current_track_index = 0
        self.next_tracks_indices = []
        self.current_volume = 0.8  # Set default volume
        self.is_playing = False
        self.history = deque(maxlen=len(self.playlist_paths) - 1)
        self.recently_played = deque(
            maxlen=len(self.playlist_paths) - 1
        )  # used to prevent duplicates for next nearest neighbors
        self.should_exit = False
        pygame.mixer.init()
        pygame.mixer.music.set_volume(self.current_volume)
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        pygame.init()

    def change_volume(self):
        try:
            new_volume = float(input("Enter new volume (0.0 to 1.0): "))
            if 0.0 <= new_volume <= 1.0:
                self.current_volume = new_volume
                pygame.mixer.music.set_volume(new_volume)
                print(f"Volume set to {new_volume}")
            else:
                print("Invalid volume. Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("Invalid input. Please enter a number between 0.0 and 1.0")

    def initialize_embeddings(self):
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.playlist_paths = []
        self.music_embeddings = []

        current_files = list(self.music_directory.rglob("*.mp3"))
        print("Initializing embeddings...")

        # Step 1: Load existing files from cache
        cached_files = []
        new_files = []
        for file in current_files:
            file_name = file.name
            file_size = os.path.getsize(file)
            cache_file = self.cache_directory / f"{file_name}_{file_size}.npz"
            if cache_file.exists():
                cached_files.append((file, cache_file))
            else:
                new_files.append(file)

        print(f"Loading {len(cached_files)} cached files...")
        for file, cache_file in tqdm(
            cached_files, desc="Loading cached files", unit="file"
        ):
            with np.load(cache_file) as data:
                embedding = data["embedding"]
            self.playlist_paths.append(file)
            self.music_embeddings.append(embedding)

        # Step 2: Embed and cache new files
        print(f"Processing {len(new_files)} new files...")
        if len(new_files) > 0:
            # Only load embedding model if new files exist
            from music_embedder import audio_embed
            # from music_embedder_mert import audio_embed

            for file in tqdm(new_files, desc="Processing new files", unit="file"):
                file_name = file.name
                file_size = os.path.getsize(file)
                cache_file = self.cache_directory / f"{file_name}_{file_size}.npz"

                embedding = audio_embed(str(file))
                np.savez_compressed(
                    cache_file, embedding=embedding, file_path=str(file)
                )

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

    def like_song(self):
        """Sample new neighbors based on current song"""
        if self.current_embedding is not None:
            print("Liked song: sampling new nearest neighbors")
            self.next_tracks_indices = self.find_nearest_embeddings()
        else:
            print("Choose a song first...")

    def next_track(self, similar=False):
        if similar and self.current_embedding is not None:
            if len(self.next_tracks_indices) == 0:
                # Fetch new similar tracks based on current track
                self.next_tracks_indices = self.find_nearest_embeddings()

            # Skip duplicate songs
            dist = np.linalg.norm(
                self.music_embeddings[self.current_track_index]
                - self.music_embeddings[self.next_tracks_indices[0]]
            )
            # Magic number to skip (nearly exact) duplicates
            if dist < 0.26:
                print(
                    "skipping duplicate:",
                    self.playlist_paths[self.next_tracks_indices[0]].name,
                    dist,
                )
                self.recently_played.append(self.next_tracks_indices[0])
                self.next_tracks_indices = self.next_tracks_indices[1:]

            # Select next track
            self.current_track_index = self.next_tracks_indices[0]
            self.next_tracks_indices = self.next_tracks_indices[1:]
            self.current_embedding = self.music_embeddings[self.current_track_index]
        else:
            self.current_track_index = random.randint(0, len(self.playlist_paths) - 1)
            self.current_embedding = self.music_embeddings[self.current_track_index]
            # Reset recently played and next tracks on random shuffle
            self.recently_played.clear()
            self.next_tracks_indices = []

        if self.is_playing:
            self.play_current_track()
        else:
            print(
                f"Next track selected: {self.playlist_paths[self.current_track_index].name}"
            )

    def find_nearest_embeddings(self, k=17):
        distances = np.linalg.norm(
            self.music_embeddings - self.current_embedding, axis=1
        )

        # Prevent looping of similar songs
        for idx in self.recently_played:
            distances[idx] = np.inf

        # Get indices of k nearest neighbors
        nearest_neighbors = np.argsort(distances)[:k]
        print("new neighbors:")
        for i in nearest_neighbors:
            print(self.playlist_paths[i].name, distances[i])
        return nearest_neighbors

    def previous_track(self):
        if len(self.history) > 1:
            self.history.pop()
            self.current_track_index = self.history.pop()
        else:
            self.current_track_index = (self.current_track_index - 1) % len(
                self.playlist_paths
            )

        if self.is_playing:
            self.play_current_track()
        else:
            print(
                f"Previous track selected: {self.playlist_paths[self.current_track_index].name}"
            )

    def shuffle_playlist(self):
        combined = list(zip(self.playlist_paths, self.music_embeddings))
        random.shuffle(combined)
        self.playlist_paths, self.music_embeddings = zip(*combined)
        self.playlist_paths = list(self.playlist_paths)
        self.music_embeddings = np.array(self.music_embeddings)
        self.recently_played.clear()
        print("Playlist shuffled.")

    @staticmethod
    def custom_scorer(query, choice, score_cutoff=0):
        if query.lower() in choice.lower():
            return 100  # Exact substring match
        else:
            return fuzz.WRatio(query, choice)  # Fuzzy match

    def fuzzy_search(self, query):
        song_names = [p.stem for p in self.playlist_paths]
        results = process.extract(
            query, song_names, scorer=self.custom_scorer, limit=10
        )
        results = [(r, s) for r, s, _i in results]
        return results

    def play_song_by_index(self, index):
        if 0 <= index < len(self.playlist_paths):
            self.current_track_index = index
            self.next_tracks_indices = []  # Reset next tracks
            self.recently_played.clear()  # Reset previous tracks queue
            self.play_current_track()
        else:
            print("Invalid song index.")

    def check_music_end(self):
        """Auto-play next song"""
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
    print("p - Play/Pause music Hotkey: Media play/pause")
    print("n - Play next track (random) Hotkey: F5")
    print("m - Play next track (similar) Hotkey: F6")
    print("b - Play previous track")
    print("l - Like song (play more similar) Hotkey: F13")
    print("f - Fuzzy search for a song by name")
    print("c - Change volume")
    print("q - Exit the player")


def handle_user_input(player):
    print_menu()
    while not player.should_exit:
        command = input("Enter command: ").lower().strip()
        if command == "p":
            player.toggle_play_pause()
        elif command == "l":
            player.like_song()
        elif command == "n":
            player.next_track(similar=False)
        elif command == "m":
            player.next_track(similar=True)
        elif command == "b":
            player.previous_track()
        elif command == "f":
            query = input("Enter search query: ")
            results = player.fuzzy_search(query)
            print("\nTop 10 matches:")
            for i, (song, score) in enumerate(results):
                print(f"{i+1}. {song} (Score: {score:.2f})")
            choice = input(
                "Enter the number of the song you want to play (or press Enter to cancel): "
            )
            if choice.isdigit() and 1 <= int(choice) <= 10:
                index = [p.name for p in player.playlist_paths].index(
                    f"{results[int(choice)-1][0]}.mp3"
                )
                player.play_song_by_index(index)
        elif command == "c":
            print("Current volume:", player.current_volume)
            player.change_volume()
        elif command == "q":
            print("Exiting Music Player.")
            player.should_exit = True
        else:
            print("Invalid command. Try again.")


def main():
    parser = argparse.ArgumentParser(description="VibeShuffle Music Player")
    parser.add_argument(
        "music_directory",
        type=str,
        help="Path to the directory containing music files (mp3)",
    )
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
        elif key == keyboard.Key.f13:
            player.like_song()
        elif key == keyboard.Key.media_play_pause:
            player.toggle_play_pause()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    player.run()


if __name__ == "__main__":
    main()
