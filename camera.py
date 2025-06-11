import cv2
from deepface import DeepFace
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CameraEmotionDetector:
    """Capture frames from webcam and detect emotions using DeepFace."""

    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def read_emotion(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None, None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, silent=True)
            emotion = result['dominant_emotion']
            probs = result['emotion']
            return emotion, probs
        except Exception as e:
            logger.debug(f"DeepFace error: {e}")
            return None, None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
