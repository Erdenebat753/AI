import logging

class ExpertSystem:
    """Simple rule-based expert system for actions."""

    def __init__(self):
        self.emotion_to_music = {
            'happy': 'happy_song.mp3',
            'sad': 'sad_song.mp3',
            'angry': 'angry_song.mp3',
            'surprise': 'surprise_song.mp3',
            'fear': 'calm_song.mp3',
            'disgust': 'relax_song.mp3',
            'neutral': 'neutral_song.mp3',
        }

    def recommend_song(self, emotion: str) -> str:
        """Return music file name for detected emotion."""
        return self.emotion_to_music.get(emotion, 'default_song.mp3')

    def execute(self, intent: str, emotion: str) -> None:
        """Execute action based on intent and emotion."""
        if intent == 'get_music':
            song = self.recommend_song(emotion)
            logging.info(f"Recommended song: {song}")
        elif intent == 'get_weather':
            logging.info("Pretending to show weather information.")
        elif intent == 'greeting':
            logging.info("Hello! Nice to meet you.")
        else:
            logging.info(f"No rule defined for intent '{intent}'.")
