import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON"]


def extract_pos_features(texts):
    n_samples = len(texts)
    n_tags = len(POS_TAGS)

    features = np.zeros((n_samples, n_tags))

    for i, doc in enumerate(nlp.pipe(texts, batch_size=64, n_process=2)):
        counts = {tag: 0 for tag in POS_TAGS}
        total = 0

        for token in doc:
            if token.is_alpha:
                total += 1
                if token.pos_ in POS_TAGS:
                    counts[token.pos_] += 1

        if total == 0:
            total = 1

        features[i] = [counts[tag] / total for tag in POS_TAGS]

    return features
