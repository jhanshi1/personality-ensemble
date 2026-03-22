import numpy as np

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
            word = word.lower()

            if emotion in EMOTIONS and association == "1":
                lexicon[emotion].add(word)

    return lexicon


def extract_emotion_features(texts, lexicon):
    n_samples = len(texts)
    n_emotions = len(EMOTIONS)

    features = np.zeros((n_samples, n_emotions))

    for i, text in enumerate(texts):
        words = text.split()

        if not words:
            continue

        word_set = set(words)
        total = len(words)

        for j, emotion in enumerate(EMOTIONS):
            count = sum(1 for w in word_set if w in lexicon[emotion])
            features[i, j] = count / total

    return features
