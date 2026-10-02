"""
Unsupervised representation learning utilities.

Pipeline:
text -> sentence embedding -> deep autoencoder -> UMAP -> HDBSCAN

No emotion labels are required by the clustering stage.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_deep_unsupervised(embeddings: np.ndarray, epochs: int = 80):
    scaler = StandardScaler()
    x_np = scaler.fit_transform(embeddings).astype("float32")
    x = torch.tensor(x_np)

    model = DeepAutoencoder(x.shape[1], latent_dim=min(64, x.shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed, _ = model(x)
        loss = loss_fn(reconstructed, x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, latent = model(x)

    latent_np = latent.cpu().numpy()

    reducer = umap.UMAP(
        n_neighbors=min(15, max(3, len(latent_np) - 1)),
        n_components=8,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(latent_np)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(3, min(8, len(reduced) // 5 or 3)),
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    return model, scaler, reducer, clusterer, labels
