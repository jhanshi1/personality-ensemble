import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON"]
def extract_pos_features(texts):
    features = []
    for doc in nlp.pipe(texts, batch_size=64):
        counts = {tag: 0 for tag in POS_TAGS}
        total = 0
        for token in doc:
            if token.pos_ in POS_TAGS:
                counts[token.pos_] += 1
            total += 1
        if total == 0:
            total = 1
        # Normalize
        row = [counts[tag] / total for tag in POS_TAGS]
        features.append(row)
    return np.array(features)
