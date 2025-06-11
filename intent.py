import json
import numpy as np
from skfuzzy import control as ctrl
from rapidfuzz import fuzz as rfuzz

class IntentRecognizer:
    """Fuzzy intent recognizer."""

    def __init__(self, intent_path: str = 'intent.json'):
        with open(intent_path, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        self._build_system()

    def _build_system(self) -> None:
        word_match = ctrl.Antecedent(np.arange(0, 101, 1), 'word_match')
        intent_score = ctrl.Consequent(np.arange(0, 101, 1), 'intent_score')
        word_match.automf(3)
        intent_score.automf(3)
        rules = [
            ctrl.Rule(word_match['good'], intent_score['good']),
            ctrl.Rule(word_match['average'], intent_score['average']),
            ctrl.Rule(word_match['poor'], intent_score['poor'])
        ]
        intent_ctrl = ctrl.ControlSystem(rules)
        self.simulator = ctrl.ControlSystemSimulation(intent_ctrl)

    def compute_scores(self, text: str):
        text = text.strip().lower()
        scores = []
        for intent in self.intents:
            max_kw_score = max(rfuzz.ratio(text, kw.lower()) for kw in intent['intent'])
            self.simulator.input['word_match'] = max_kw_score
            self.simulator.compute()
            scores.append(self.simulator.output['intent_score'])
        return scores

    def best_intent(self, scores, threshold: float = 60.0) -> str:
        best_score = max(scores)
        if best_score > threshold:
            idx = scores.index(best_score)
            return self.intents[idx]['name']
        return 'unknown'
