# SMH Cheerful — Human Sentiment & Emotion Analyzer

> **SMH Cheerful** is an AI-powered human emotion analyzer that reads natural-language input and estimates the emotional signals behind it.

Example:

> "I failed my test and I feel like I disappointed everyone."

The system can surface emotions such as **sadness, guilt, anxiety, frustration, shame, fear, loneliness, disappointment, hope, joy, anger, disgust, surprise, gratitude, and love**.

## What makes this project different?

The primary emotion-discovery pipeline is intentionally **unsupervised**:

1. A Hugging Face `sentence-transformers/all-MiniLM-L6-v2` encoder converts text into semantic embeddings.
2. A deep autoencoder learns a compressed latent representation without emotion labels.
3. UMAP reduces the learned latent space for structure discovery.
4. HDBSCAN discovers dense, naturally occurring language clusters.
5. Emotion prototype embeddings are used only to semantically align discovered clusters with human-readable emotion names.
6. Cosine similarity produces a multi-emotion profile instead of forcing every sentence into one class.

This is a portfolio/demo architecture for **unsupervised emotion discovery**, not a clinical or psychological diagnostic system.

## Tech stack

### Frontend
- React + Vite
- CSS animations
- Responsive glassmorphism UI
- No paid UI framework required

### Backend
- Python
- FastAPI
- Sentence Transformers
- PyTorch
- scikit-learn
- UMAP
- HDBSCAN

### DevOps
- Docker
- Docker Compose
- GitHub Actions

## Project structure

```text
smh-cheerful/
├── .github/
│   └── workflows/
│       └── docker-build.yml
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── emotion_engine.py
│   │   └── ml/
│   │       └── unsupervised.py
│   ├── data/
│   │   └── sample_corpus.txt
│   ├── scripts/
│   │   └── train_unsupervised.py
│   ├── artifacts/
│   │   └── .gitkeep
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Intro.jsx
│   │   │   ├── Analyzer.jsx
│   │   │   └── EmotionOrb.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Run locally

### Option A — Docker

```bash
docker compose up --build
```

Frontend:
`http://localhost:5173`

Backend:
`http://localhost:8000`

Swagger API:
`http://localhost:8000/docs`

### Option B — Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Option C — Frontend

```bash
cd frontend
npm install
npm run dev
```

## Train the unsupervised representation

From the `backend` directory:

```bash
python scripts/train_unsupervised.py
```

The script creates a latent representation using a PyTorch autoencoder and then applies UMAP + HDBSCAN to discover structure in the unlabeled corpus.

For a stronger experiment, replace `backend/data/sample_corpus.txt` with a much larger unlabeled corpus.

## API

### POST `/api/analyze`

```json
{
  "text": "I failed my test and I feel terrible about disappointing my parents."
}
```

Response:

```json
{
  "primary_emotion": "sadness",
  "confidence": 0.82,
  "emotions": [
    {"emotion": "sadness", "score": 0.82},
    {"emotion": "guilt", "score": 0.76},
    {"emotion": "anxiety", "score": 0.59}
  ],
  "cluster": 3,
  "explanation": "The language is semantically close to sadness, guilt and anxiety."
}
```

## Important ML note

A truly unsupervised system does not magically know that a cluster means "sadness". Clustering discovers structure; **semantic prototype alignment** gives those clusters human-readable emotion names. This project keeps that distinction explicit.

A supervised Hugging Face emotion checkpoint can be used as a baseline for comparison, but it is **not** the primary learning method in this project.

## Disclaimer

SMH Cheerful estimates linguistic emotion signals. It should not be used for medical diagnosis, mental-health diagnosis, crisis assessment, hiring decisions, or other high-stakes decisions.
