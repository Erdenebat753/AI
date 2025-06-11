import matplotlib.pyplot as plt

class LivePlot:
    def __init__(self, intent_names, emotion_names):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.intent_names = intent_names
        self.emotion_names = emotion_names

    def update(self, intent_scores, selected_intent, emotion_probs, dominant_emotion):
        # Intent plot
        self.ax1.clear()
        self.ax1.bar(self.intent_names, intent_scores, color=['green' if n == selected_intent else 'blue' for n in self.intent_names])
        self.ax1.set_ylim(0, 100)
        self.ax1.set_ylabel('Fuzzy Intent Score')
        self.ax1.set_title(f'Selected Intent: {selected_intent}')
        # Emotion plot
        self.ax2.clear()
        values = [emotion_probs.get(e, 0) for e in self.emotion_names]
        self.ax2.bar(self.emotion_names, values, color=['green' if e == dominant_emotion else 'blue' for e in self.emotion_names])
        self.ax2.set_ylim(0, 100)
        self.ax2.set_ylabel('Probability (%)')
        self.ax2.set_title(f'Detected Emotion: {dominant_emotion}')
        plt.tight_layout()
        plt.pause(0.1)

    def keep_open(self):
        plt.ioff()
        plt.show() 
