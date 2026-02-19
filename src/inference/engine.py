import os
import torch
import numpy as np
import joblib
from transformers import BertTokenizer
from scipy.sparse import csr_matrix, hstack

from src.models.dl.bigru import BiGRUClassifier
from src.models.dl.bilstm_bert import BiLSTMBERTClassifier
from src.models.dl.cnn_bert import CNNBERTClassifier
from src.models.dl.ft_bert import FineTunedBERT
from src.data.dl_preprocess import encode_texts, pad_sequences

from src.features.pos_features import extract_pos_features
from src.features.emotion_features import (
    load_nrc_lexicon,
    extract_emotion_features
)


class PersonalityEngine:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.TRAITS = ["OPN", "CON", "EXT", "AGR", "NEU"]

        self.MAX_LEN_RNN = 50
        self.MAX_LEN_BERT = 64

        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )

        self.CLASSIC_DIR = os.path.join(BASE_DIR, "artifacts", "classic")
        self.DEEP_DIR = os.path.join(BASE_DIR, "artifacts", "deep")
        self.DL_DATA_DIR = os.path.join(BASE_DIR, "artifacts", "dl_data")
        self.ENSEMBLE_DIR = os.path.join(BASE_DIR, "artifacts", "ensemble")

        self._load_classical()
        self._load_deep()
        self._load_xgboost()

        

    # -------------------------------------------------
    # LOADERS
    # -------------------------------------------------

    def _load_classical(self):

        # Load NRC lexicon once
        self.nrc = load_nrc_lexicon("data/external/NRC-Emotion-Lexicon.txt")

        # Logistic Regression
        self.lr_model = joblib.load(
            os.path.join(self.CLASSIC_DIR, "lr_model.pkl")
        )
        self.lr_word_vectorizer = joblib.load(
            os.path.join(self.CLASSIC_DIR, "word_vectorizer.pkl")
        )
        self.lr_char_vectorizer = joblib.load(
            os.path.join(self.CLASSIC_DIR, "char_vectorizer.pkl")
        )

        # SVM
        self.svm_model = joblib.load(
            os.path.join(self.CLASSIC_DIR, "svm_model.pkl")
        )
        self.svm_word_vectorizer = joblib.load(
            os.path.join(self.CLASSIC_DIR, "svm_word_vectorizer.pkl")
        )
        self.svm_char_vectorizer = joblib.load(
            os.path.join(self.CLASSIC_DIR, "svm_char_vectorizer.pkl")
        )
        self.svm_scaler = joblib.load(
            os.path.join(self.CLASSIC_DIR, "svm_scaler.pkl")
        )

    def _load_deep(self):

        self.word2idx = np.load(
            os.path.join(self.DL_DATA_DIR, "word2idx.npy"),
            allow_pickle=True
        ).item()

        self.embedding_matrix = np.load(
            os.path.join(self.DL_DATA_DIR, "glove_embedding_matrix.npy")
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # BiGRU
        bigru_ckpt = torch.load(
            os.path.join(self.DEEP_DIR, "bigru_model.pt"),
            map_location=self.device
        )
        self.bigru = BiGRUClassifier(self.embedding_matrix)
        self.bigru.load_state_dict(bigru_ckpt["model_state_dict"])
        self.bigru.to(self.device)
        self.bigru.eval()

        # BiLSTM-BERT
        bilstm_ckpt = torch.load(
            os.path.join(self.DEEP_DIR, "bilstm_model.pt"),
            map_location=self.device
        )
        self.bilstm = BiLSTMBERTClassifier()
        self.bilstm.load_state_dict(bilstm_ckpt["model_state_dict"])
        self.bilstm.to(self.device)
        self.bilstm.eval()

        # CNN-BERT
        cnn_ckpt = torch.load(
            os.path.join(self.DEEP_DIR, "cnn_model.pt"),
            map_location=self.device
        )
        self.cnn = CNNBERTClassifier()
        self.cnn.load_state_dict(cnn_ckpt["model_state_dict"])
        self.cnn.to(self.device)
        self.cnn.eval()

        # FT-BERT
        ftbert_ckpt = torch.load(
            os.path.join(self.DEEP_DIR, "ft_bert_model.pt"),
            map_location=self.device
        )
        self.ftbert = FineTunedBERT()
        self.ftbert.load_state_dict(ftbert_ckpt["model_state_dict"])
        self.ftbert.to(self.device)
        self.ftbert.eval()

    def _load_xgboost(self):

        meta_path = os.path.join(self.ENSEMBLE_DIR, "xgboost_meta.pkl")

        print("Checking XGB path:", meta_path)
        print("Exists?", os.path.exists(meta_path))

        if os.path.exists(meta_path):
            self.xgb_model = joblib.load(meta_path)
            print("XGB model loaded successfully.")
            print("Type:", type(self.xgb_model))
        else:
            self.xgb_model = None
            print("XGB model set to None.")


    # -------------------------------------------------
    # CLASSICAL FEATURE BUILDER
    # -------------------------------------------------

    def _build_classical_features(self, text, word_vec, char_vec):

        word_feats = word_vec.transform([text])
        char_feats = char_vec.transform([text])

        pos_feats = csr_matrix(extract_pos_features([text]))
        emo_feats = csr_matrix(extract_emotion_features([text], self.nrc))

        X = hstack([word_feats, char_feats, pos_feats, emo_feats])
        return X

    # -------------------------------------------------
    # INTERNAL CLASSICAL PREDICTION
    # -------------------------------------------------

    def _predict_lr_internal(self, text):

        X = self._build_classical_features(
            text,
            self.lr_word_vectorizer,
            self.lr_char_vectorizer
        )

        probs = self.lr_model.predict_proba(X)[0]
        return probs

    def _predict_svm_internal(self, text):

        X = self._build_classical_features(
            text,
            self.svm_word_vectorizer,
            self.svm_char_vectorizer
        )

        X = self.svm_scaler.transform(X)

        probs = self.svm_model.predict_proba(X)[0]
        return probs

    # -------------------------------------------------
    # INTERNAL DEEP PREDICTION
    # -------------------------------------------------

    def _predict_glove(self, model, text):

        encoded = encode_texts([text], self.word2idx)
        padded = pad_sequences(encoded, self.MAX_LEN_RNN)

        inputs = torch.tensor(padded, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return probs

    def _predict_bert(self, model, text):

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.MAX_LEN_BERT,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        return probs

    # -------------------------------------------------
    # FORMAT OUTPUT
    # -------------------------------------------------

    def _format_output(self, probs):

        return {
            trait: float(prob)
            for trait, prob in zip(self.TRAITS, probs)
        }

    # -------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------

    def predict_lr(self, text):
        return self._format_output(self._predict_lr_internal(text))

    def predict_svm(self, text):
        return self._format_output(self._predict_svm_internal(text))

    def predict_bigru(self, text):
        return self._format_output(self._predict_glove(self.bigru, text))

    def predict_cnn_bert(self, text):
        return self._format_output(self._predict_bert(self.cnn, text))

    def predict_bilstm_bert(self, text):
        return self._format_output(self._predict_bert(self.bilstm, text))

    def predict_ftbert(self, text):
        return self._format_output(self._predict_bert(self.ftbert, text))

    def predict_mean(self, text):
        outputs = []
        outputs.append(self._predict_lr_internal(text))
        outputs.append(self._predict_svm_internal(text))
        outputs.append(self._predict_glove(self.bigru, text))
        outputs.append(self._predict_bert(self.cnn, text))
        outputs.append(self._predict_bert(self.bilstm, text))
        outputs.append(self._predict_bert(self.ftbert, text))

        # Convert to numpy array (6, 5)
        outputs = np.array(outputs)

        # Mean across models
        final = np.mean(outputs, axis=0)

        return self._format_output(final)
 

    def predict_xgboost(self, text):

        if self.xgb_model is None:
            raise ValueError("XGBoost meta model not found.")

        lr = self._predict_lr_internal(text)
        svm = self._predict_svm_internal(text)
        bigru = self._predict_glove(self.bigru, text)
        cnn = self._predict_bert(self.cnn, text)
        bilstm = self._predict_bert(self.bilstm, text)
        ftbert = self._predict_bert(self.ftbert, text)

        # Base 30 features
        base_features = np.concatenate(
            [lr, svm, bigru, cnn, bilstm, ftbert]
        )

        # ----- ADD INTERACTION FEATURES (same as training) -----

        ftbert_probs = ftbert  # last 5 values in training logic

        interactions = []
        for i in range(5):
            for j in range(i + 1, 5):
                interactions.append(ftbert_probs[i] * ftbert_probs[j])

        interactions = np.array(interactions)

        # Final 40 features
        features = np.concatenate([base_features, interactions]).reshape(1, -1)

        # Extract probabilities correctly
        probs = np.column_stack([
            est.predict_proba(features)[:, 1]
            for est in self.xgb_model.estimators_
        ])

        return self._format_output(probs[0])
    def predict_all_models(self, text):

        results = {}

        # Classical
        results["lr"] = self.predict_lr(text)
        results["svm"] = self.predict_svm(text)

        # Deep
        results["bigru"] = self.predict_bigru(text)
        results["cnn_bert"] = self.predict_cnn_bert(text)
        results["bilstm_bert"] = self.predict_bilstm_bert(text)
        results["ftbert"] = self.predict_ftbert(text)

        # Ensemble
        results["mean"] = self.predict_mean(text)

        try:
            results["xgboost"] = self.predict_xgboost(text)
        except Exception as e:
            results["xgboost"] = {"error": str(e)}

        return results


