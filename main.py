import logging
import threading
import time

from camera import CameraEmotionDetector
from intent import IntentRecognizer
from graphic import LivePlot
from expert import ExpertSystem

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def emotion_worker(detector: CameraEmotionDetector, shared: dict) -> None:
    """Thread worker to continuously read emotions from the camera."""
    while not shared['stop']:
        emotion, probs = detector.read_emotion()
        if emotion:
            shared['emotion'] = emotion
            shared['emotion_probs'] = probs
        if shared['stop']:
            break
    detector.release()


def main() -> None:
    detector = CameraEmotionDetector()
    intent_recognizer = IntentRecognizer()
    expert = ExpertSystem()

    shared = {
        'emotion': 'neutral',
        'emotion_probs': {k: 0 for k in expert.emotion_to_music.keys()},
        'stop': False,
    }

    t = threading.Thread(target=emotion_worker, args=(detector, shared), daemon=True)
    t.start()

    intent_names = [intent['name'] for intent in intent_recognizer.intents]
    emotion_names = list(expert.emotion_to_music.keys())
    plotter = LivePlot(intent_names, emotion_names)
    scores = [0] * len(intent_names)
    selected = 'unknown'

    logger.info("System started. Type 'exit' to quit.")
    try:
        while True:
            text = input('Text (or "exit" to quit): ')
            if text.strip().lower() == 'exit':
                break

            scores = intent_recognizer.compute_scores(text)
            selected = intent_recognizer.best_intent(scores)
            plotter.update(scores, selected, shared['emotion_probs'], shared['emotion'])
            time.sleep(0.1)
            expert.execute(selected, shared['emotion'])
    finally:
        shared['stop'] = True
        t.join()
        plotter.update(scores, selected, shared['emotion_probs'], shared['emotion'])
        plotter.keep_open()


if __name__ == '__main__':
    main()
