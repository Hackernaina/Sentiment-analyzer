from __future__ import annotations

import os
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .ml.unsupervised import train_deep_unsupervised


EMOTIONS = {
    "joy": "I feel happy, cheerful, delighted, excited and full of positive energy.",
    "sadness": "I feel sad, hurt, low, heartbroken, unhappy and emotionally down.",
    "anger": "I feel angry, irritated, furious, annoyed, resentful and upset.",
    "fear": "I feel afraid, scared, nervous, threatened and worried about danger.",
    "surprise": "I feel surprised, shocked, amazed, unexpected and stunned.",
    "disgust": "I feel disgusted, repulsed, uncomfortable and strongly put off.",
    "love": "I feel love, affection, closeness, warmth and deep care for someone.",
    "gratitude": "I feel thankful, grateful, appreciative and fortunate.",
    "guilt": "I feel guilty, responsible, regretful and bad about something I did.",
    "shame": "I feel ashamed, embarrassed, exposed, humiliated or unworthy.",
    "anxiety": "I feel anxious, overwhelmed, restless, uncertain and unable to stop worrying.",
    "hope": "I feel hopeful, optimistic, encouraged and believe things can improve.",
    "frustration": "I feel frustrated, stuck, blocked, disappointed and unable to make progress.",
    "loneliness": "I feel lonely, isolated, disconnected and like I have nobody with me.",
    "disappointment": "I feel disappointed because reality did not meet my expectations.",
}


class EmotionEngine:
    def __init__(self):
        model_name = os.getenv(
            "MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.encoder = SentenceTransformer(model_name)
        self.emotion_names = list(EMOTIONS.keys())
        self.prototype_texts = list(EMOTIONS.values())
        self.prototype_embeddings = self.encoder.encode(
            self.prototype_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.clusterer = None
        self.reducer = None
        self.autoencoder = None
        self.scaler = None

        # A compact unlabeled corpus gives the application a real
        # unsupervised discovery stage on startup.
        corpus_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "sample_corpus.txt",
        )
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = [line.strip() for line in f if line.strip()]

            if len(corpus) >= 15:
                embeddings = self.encoder.encode(
                    corpus,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                (
                    self.autoencoder,
                    self.scaler,
                    self.reducer,
                    self.clusterer,
                    self.labels,
                ) = train_deep_unsupervised(embeddings, epochs=30)
        except Exception:
            # The API still works through semantic prototype similarity
            # if clustering dependencies/data are unavailable.
            self.clusterer = None

    def analyze(self, text: str) -> dict[str, Any]:
        embedding = self.encoder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = cosine_similarity(embedding, self.prototype_embeddings)[0]

        # Convert cosine similarity into a positive, readable score.
        shifted = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)

        ranked = sorted(
            zip(self.emotion_names, shifted.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        top = ranked[:5]
        primary, primary_score = top[0]

        cluster = None
        if self.clusterer is not None and self.reducer is not None:
            try:
                # HDBSCAN prediction requires the same reduced space.
                # We intentionally expose cluster discovery as a supporting
                # signal, while emotion names remain prototype-aligned.
                from hdbscan.prediction import approximate_predict

                latent = self.autoencoder.encoder(
                    __import__("torch").tensor(
                        self.scaler.transform(embedding.astype("float32"))
                    )
                ).detach().cpu().numpy()

                reduced = self.reducer.transform(latent)
                cluster, _ = approximate_predict(self.clusterer, reduced)
                cluster = int(cluster[0])
            except Exception:
                cluster = None

        explanation = (
            "The language is semantically closest to "
            + ", ".join(name for name, _ in top[:3])
            + ". The scores represent linguistic similarity, not a clinical diagnosis."
        )

        return {
            "primary_emotion": primary,
            "confidence": round(float(primary_score), 3),
            "emotions": [
                {"emotion": name, "score": round(float(score), 3)}
                for name, score in top
            ],
            "cluster": cluster,
            "explanation": explanation,
        }
