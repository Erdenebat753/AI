import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from rapidfuzz import fuzz as rfuzz

# Load intents from intent.json
with open('intent.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Fuzzy logic system for intent recognition
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
