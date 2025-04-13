import cv2
import mediapipe as mp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import threading
from spotipy.exceptions import SpotifyException

# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="your_client_id",
    client_secret="your_client_secret",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-library-read user-read-playback-state user-modify-playback-state",
    requests_timeout=20
))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variables
current_frame = None
current_volume = 50  # Initial volume
frame_lock = threading.Lock()

# Cooldowns (in seconds)
gesture_cooldowns = {
    'play_pause': 1.5,
    'next': 2,
    'previous': 2,
    'volume_up': 1,
    'volume_down': 1
}
last_triggered = {k: 0 for k in gesture_cooldowns.keys()}

def cooldown_expired(gesture):
    return time.time() - last_triggered[gesture] > gesture_cooldowns[gesture]

def capture_frame():
    global current_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        with frame_lock:
            current_frame = frame
        time.sleep(0.01)

    cap.release()

def process_frame():
    global current_frame, current_volume

    cv2.namedWindow("Hand Tracker 🎵", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracker 🎵", 800, 600)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            if current_frame is not None:
                with frame_lock:
                    frame = current_frame.copy()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

                        try:
                            # Play/Pause gesture
                            if distance < 0.05 and cooldown_expired('play_pause'):
                                playback = sp.current_playback()
                                if playback and playback['is_playing']:
                                    sp.pause_playback()
                                else:
                                    sp.start_playback()
                                last_triggered['play_pause'] = time.time()

                            # Swipe Left = Previous
                            if index_tip.x < 0.25 and cooldown_expired('previous'):
                                sp.previous_track()
                                last_triggered['previous'] = time.time()

                            # Swipe Right = Next
                            if index_tip.x > 0.75 and cooldown_expired('next'):
                                sp.next_track()
                                last_triggered['next'] = time.time()

                            # Volume Up
                            if (thumb_tip.y < middle_tip.y and index_tip.y < middle_tip.y and cooldown_expired('volume_up')):
                                current_volume = min(current_volume + 10, 100)
                                sp.volume(current_volume)
                                last_triggered['volume_up'] = time.time()

                            # Volume Down
                            if (thumb_tip.y > middle_tip.y and index_tip.y > middle_tip.y and cooldown_expired('volume_down')):
                                current_volume = max(current_volume - 10, 0)
                                sp.volume(current_volume)
                                last_triggered['volume_down'] = time.time()

                        except SpotifyException as e:
                            print("Spotify API Error:", e)

                cv2.imshow("Hand Tracker 🎵", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

                time.sleep(0.01)

    cv2.destroyAllWindows()

# Start threads
threading.Thread(target=capture_frame, daemon=True).start()
threading.Thread(target=process_frame, daemon=True).start()

# Keep alive
while True:
    time.sleep(1)
