from src.inference.engine import PersonalityEngine

engine = PersonalityEngine()

text = "I enjoy meeting new people and exploring creative ideas."

print("FT-BERT:", engine.predict_ftbert(text))
print("LR:", engine.predict_lr(text))
print("Weighted:", engine.predict_weighted(text))
print("XGB:", engine.predict_xgboost(text))
