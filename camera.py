import cv2
from deepface import DeepFace
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable tqdm progress bars
os.environ['TQDM_DISABLE'] = '1'

def get_emotion(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True,  # Disable all progress bars and warnings
            detector_backend='opencv',  # Use OpenCV for faster detection
            prog_bar=False  # Explicitly disable progress bar
        )
        emotion = result['dominant_emotion']
        emotion_probs = result['emotion']
        return emotion, emotion_probs
    except Exception as e:
        logger.debug(f"DeepFace error: {e}")
        return None, None 