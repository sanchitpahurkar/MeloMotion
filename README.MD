# **MeloMotion -** Hand-tracking tool to control spotify actions

This project allows you to control Spotify playback using hand gestures. By using **MediaPipe** for hand tracking and **Spotipy** for interacting with the Spotify API, you can control various aspects of Spotify such as play/pause, skipping to the next/previous song, and adjusting the volume.

## Features

- **Play/Pause Spotify music**
    [![Play/Pause Demonstration](https://img.youtube.com/vi/kdEsPDuot1I/0.jpg)](https://www.youtube.com/watch?v=kdEsPDuot1I)


- **Skip to the next or previous song**
    [![Prev/Next seek Demonstration](https://img.youtube.com/vi/8KVYpTxMSHE/0.jpg)](https://www.youtube.com/watch?v=8KVYpTxMSHE)

- **Adjust the volume**
    [![Prev/Next seek Demonstration](https://img.youtube.com/vi/lzhi4fmYxV4/0.jpg)](https://www.youtube.com/watch?v=lzhi4fmYxV4)

## **Install the required dependencies by running:**

```bash
pip install -r requirements.txt
```

## **Setup**
1. Spotify API Credentials:

	a. You'll need to create a Spotify Developer account and generate API credentials (client ID, client secret, and redirect URI).

    b. Replace the credentials in the code with your own to authenticate with the Spotify API.

2. Run the application:

    a. The app will open a webcam feed and start tracking your hands.

    b. Perform the gestures to control Spotify:

       i. Play/Pause: bring the index finger and thumb close to play/pause the song.

        ii. Next/Previous Song: Index finger towards left plays previous song, and towards right plays the next song

        iii. Volume Up/Down: Open palm and closed fist decreases/increases the volume
