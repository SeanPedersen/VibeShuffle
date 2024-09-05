# VibeShuffle: smart music shuffle

- embed all songs into N vectors
- play a random song
- full play is positive signal
- skip is negative signal
- positive: play more similar songs
- negative: play a random song

## Setup

Download [Pixi](https://pixi.sh/latest/#installation)

- Install: ```$ pixi install```
- Activate env: ```$ pixi shell```
- Run: ```$ python main.py```

## References

- Song Embedding Model: <https://github.com/fschmid56/EfficientAT>

## Quick way to fetch mp3 files

Copy all your liked Spotify songs (ctrl+a, ctrl+c) into a new playlist. Copy the playlist URL and use a tool like [SpotDL](https://github.com/spotDL/spotify-downloader) to download the mp3 files.
