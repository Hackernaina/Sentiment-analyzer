"""
Train the unsupervised representation-learning stage.

This script intentionally uses NO emotion labels.
It learns latent structure from raw text embeddings with:
- deep autoencoder
- UMAP
- HDBSCAN

For a serious experiment, provide thousands of unlabeled sentences.
"""

import os
import pickle

from sentence_transformers import SentenceTransformer

from app.ml.unsupervised import train_deep_unsupervised


ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS = os.path.join(ROOT, "data", "sample_corpus.txt")
ARTIFACTS = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)


def main():
    with open(CORPUS, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    model, scaler, reducer, clusterer, labels = train_deep_unsupervised(
        embeddings, epochs=100
    )

    with open(os.path.join(ARTIFACTS, "clustering.pkl"), "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "reducer": reducer,
                "clusterer": clusterer,
                "labels": labels,
            },
            f,
        )

    model.save_state_dict(os.path.join(ARTIFACTS, "autoencoder.pt"))

    print("Training complete.")
    print("Clusters discovered:", sorted(set(labels.tolist())))


if __name__ == "__main__":
    main()
