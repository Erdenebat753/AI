import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from rapidfuzz import fuzz as rfuzz
import matplotlib.pyplot as plt
import cv2
from deepface import DeepFace
import threading
import queue
import time
import logging
import os
from camera import get_emotion
from intent import get_intent_scores, get_best_intent
from graphic import LivePlot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Only show the message, not the logger name
)
logger = logging.getLogger(__name__)

# Disable tqdm progress bars
os.environ['TQDM_DISABLE'] = '1'

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load intents
with open('intent.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Emotion to music mapping (example)
emotion_music = {
    'happy': 'happy_song.mp3',
    'sad': 'sad_song.mp3',
    'angry': 'angry_song.mp3',
    'surprise': 'surprise_song.mp3',
    'fear': 'calm_song.mp3',
    'disgust': 'relax_song.mp3',
    'neutral': 'neutral_song.mp3'
}

# Fuzzy logic for intent
word_match = ctrl.Antecedent(np.arange(0, 101, 1), 'word_match')
intent_score = ctrl.Consequent(np.arange(0, 101, 1), 'intent_score')
word_match.automf(3)
intent_score.automf(3)
rule1 = ctrl.Rule(word_match['good'], intent_score['good'])
rule2 = ctrl.Rule(word_match['average'], intent_score['average'])
rule3 = ctrl.Rule(word_match['poor'], intent_score['poor'])
intent_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
intent_sim = ctrl.ControlSystemSimulation(intent_ctrl)

def get_intent_scores(text):
    text = text.strip().lower()
    scores = []
    for intent in intents:
        max_keyword_score = max(rfuzz.ratio(text, kw.lower()) for kw in intent['intent'])
        intent_sim.input['word_match'] = max_keyword_score
        intent_sim.compute()
        score = intent_sim.output['intent_score']
        scores.append(score)
    return scores

def get_best_intent(scores):
    best_score = max(scores)
    if best_score > 60:
        idx = scores.index(best_score)
        return intents[idx]['name']
    return 'unknown'

def get_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result['dominant_emotion']
        emotion_probs = result['emotion']
        return emotion, emotion_probs
    except Exception as e:
        return None, None

def emotion_thread_fn(shared):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        shared['stop'] = True
        return

    frame_count = 0
    while not shared['stop']:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            continue

        # Process every 5th frame to reduce CPU usage
        if frame_count % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emotion, emotion_probs = get_emotion(rgb_frame)
            if emotion_probs:
                shared['emotion'] = emotion
                shared['emotion_probs'] = emotion_probs
        frame_count += 1

        shared['frame'] = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared['stop'] = True
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    try:
        # Shared state for emotion
        shared = {
            'emotion': 'neutral',
            'emotion_probs': {k: 0 for k in emotion_music.keys()},
            'stop': False,
            'frame': None
        }

        # Start emotion thread
        t = threading.Thread(target=emotion_thread_fn, args=(shared,), daemon=True)
        t.start()

        intent_names = [intent['name'] for intent in intents]
        emotion_names = list(emotion_music.keys())
        plotter = LivePlot(intent_names, emotion_names)
        intent_scores = [0] * len(intent_names)
        selected_intent = 'unknown'

        logger.info("System started. Type 'exit' to quit.")
        
        while True:
            text = input('Text (or "exit" to quit): ')
            if text.strip().lower() == 'exit':
                shared['stop'] = True
                t.join()
                break

            intent_scores = get_intent_scores(text)
            selected_intent = get_best_intent(intent_scores)
            
            # Show both intent and latest emotion (live update)
            plotter.update(intent_scores, selected_intent, shared['emotion_probs'], shared['emotion'])
            time.sleep(0.1)

            if selected_intent == 'get_music':
                # Try to get emotion up to 3 times
                max_retries = 3
                retry_count = 0
                emotion_detected = False

                while retry_count < max_retries and not emotion_detected:
                    if shared['emotion'] != 'neutral' or any(v > 0 for v in shared['emotion_probs'].values()):
                        emotion_detected = True
                    else:
                        logger.info(f"Retrying emotion detection... Attempt {retry_count + 1}/{max_retries}")
                        time.sleep(1)  # Wait for 1 second before next attempt
                        retry_count += 1

                if not emotion_detected:
                    logger.warning("Could not detect emotion after 3 attempts, using neutral emotion")

                break

        # Final update and cleanup
        shared['stop'] = True
        t.join()
        plotter.update(intent_scores, selected_intent, shared['emotion_probs'], shared['emotion'])
        plotter.keep_open()

        # Print recommended song
        recommended_song = emotion_music.get(shared['emotion'], 'default_song.mp3')
        logger.info(f"Recommended song: {recommended_song}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        shared['stop'] = True
        if t.is_alive():
            t.join()

if __name__ == '__main__':
    main() 