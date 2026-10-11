from sentence_transformers import SentenceTransformer
from pathlib import Path
from joblib import load

SENTENCE_TRANSFORMER_PATH = Path(__file__).parent / 'sentence_transformer.model'
CLASSIFIER_PATH = Path(__file__).parent / 'classifier.joblib'

class_map = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

class SentimentAnalysis:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer(
            str(SENTENCE_TRANSFORMER_PATH),
            local_files_only=True,
        )
        self.classifier_path = CLASSIFIER_PATH

    def predict(self, text: str):
        embedding = self.sentence_transformer.encode([text])
        classifier = load(self.classifier_path)

        return class_map[classifier.predict(embedding)[0]]
