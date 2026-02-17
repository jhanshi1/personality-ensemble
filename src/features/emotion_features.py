import numpy as np
import os

EMOTIONS = [
    "anger",
    "fear",
    "sadness",
    "joy",
    "trust",
    "disgust",
    "surprise",
    "anticipation"
]

def load_nrc_lexicon(path):
    lexicon = {emotion: set() for emotion in EMOTIONS}
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, association = parts
            if emotion in EMOTIONS and association == "1":
                lexicon[emotion].add(word)
    return lexicon



def extract_emotion_features(texts, lexicon):
    features = []
    for text in texts:
        words = text.split()
        total = len(words)
        if total == 0:
            total = 1
        row = []
        for emotion in EMOTIONS:
            count = sum(1 for w in words if w in lexicon[emotion])
            row.append(count / total)
        features.append(row)
    return np.array(features)
