# VibeShuffle: smart music shuffle

- Embeds all songs (mp3 files) using neural net (results are cached)
- Play a random song or make a fuzzy search
- Next similar: play more similar songs
- Next shuffle: play a random song
- Default: play similar songs
- Like song: play more similar songs (sample new nearest neighbors)
- Hotkeys: play/pause media, F5: next shuffle, F6: next similar, F13: like song

## Setup

Download and install [Pixi](https://pixi.sh/latest/#installation)

- Change dir to vibe-shuffle: ```$ cd vibe-shuffle/```
- Install dependencies: ```$ pixi install```
- Activate environment: ```$ pixi shell```
- Run app: ```$ python main.py path/to/your/music```

## References

- Song Embedding Model: <https://github.com/fschmid56/EfficientAT>

## Quick way to fetch mp3 files

Copy all your liked Spotify songs (ctrl+a, ctrl+c) into a new playlist. Copy the playlist URL and use a tool like [SpotDL](https://github.com/spotDL/spotify-downloader) to download the mp3 files.
