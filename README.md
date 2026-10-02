
# 🌸 SMH Cheerful — Human Sentiment & Emotion Analyzer

> **An AI-powered emotion analysis system that looks beyond simple positive/negative sentiment to discover the emotional signals hidden inside human language.**

**SMH Cheerful** analyzes a sentence such as:

> *"I failed my test and I feel like I disappointed everyone."*

Instead of forcing the sentence into a single sentiment such as *positive* or *negative*, the system identifies a **multi-emotion profile** such as:

**Sadness · Guilt · Disappointment · Anxiety · Frustration**

The project combines **Sentence Transformers, semantic similarity, PyTorch representation learning, UMAP, and HDBSCAN** to create an emotion-analysis pipeline with an unsupervised discovery component.

---

## ✨ Why SMH Cheerful?

Traditional sentiment analysis often reduces human language to:

```text
Positive
Negative
Neutral
```

But real human emotions are rarely that simple.

For example:

> **"I got the internship, but I'm terrified that I'm not good enough."**

This sentence can contain:

* 🎉 Joy
* 😰 Anxiety
* 😨 Fear
* 🌱 Hope

SMH Cheerful therefore produces an **emotional map** rather than treating emotion as a single classification label.

---

# 🧠 Key Features

### 💭 Multi-Emotion Detection

The system evaluates the input against **15 emotional dimensions**:

| Emotion           | Emotion       |
| ----------------- | ------------- |
| 😊 Joy            | 😢 Sadness    |
| 😡 Anger          | 😨 Fear       |
| 😲 Surprise       | 🤢 Disgust    |
| ❤️ Love           | 🙏 Gratitude  |
| 😔 Guilt          | 😳 Shame      |
| 😰 Anxiety        | 🌱 Hope       |
| 😤 Frustration    | 🥀 Loneliness |
| 💔 Disappointment |               |

The output contains the strongest emotional signals rather than only one label.

---

### 🤖 Semantic Emotion Analysis

SMH Cheerful uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to transform natural language into high-dimensional semantic embeddings.

For example:

```text
"I failed my exam"
        ↓
Semantic Embedding
        ↓
[0.12, -0.31, 0.87, ...]
```

The embedding is then compared against semantic representations of predefined emotional concepts.

---

### 🧬 Unsupervised Representation Learning

The project includes an unsupervised machine-learning pipeline:

```text
Text
  ↓
Sentence Transformer
  ↓
Semantic Embedding
  ↓
StandardScaler
  ↓
Deep Autoencoder
  ↓
Latent Representation
  ↓
UMAP
  ↓
HDBSCAN
  ↓
Discovered Language Clusters
```

No emotion labels are required during the clustering stage.

---

### 🎯 Semantic Prototype Alignment

Clustering alone does not automatically know that:

```text
Cluster 4 = Sadness
```

Therefore, SMH Cheerful uses semantic emotion prototypes to provide human-readable emotion names.

The system maintains prototype descriptions such as:

```text
"I feel sad, hurt, low, heartbroken,
unhappy and emotionally down."
```

These prototype embeddings are compared with the user's input using **cosine similarity**.

This allows the system to produce interpretable emotion signals.

---

### 📊 Emotion Confidence Visualization

The frontend displays:

* Primary emotion
* Semantic confidence
* Top emotional signals
* Emotion strength bars
* AI-generated explanation
* Optional discovered cluster

Example:

```text
PRIMARY SIGNAL

       SADNESS

██████████████████░░  82%

Sadness       ██████████████████ 82%
Guilt         ███████████████    76%
Anxiety       ███████████        59%
Frustration   ██████████         51%
Disappointment█████████          47%
```

---

### 🌸 Modern AI Interface

The frontend is built with React and includes:

* Glassmorphism-inspired interface
* Animated emotion orb
* Aurora-style background effects
* Responsive layout
* Interactive text analyzer
* Loading states
* API error handling
* Emotion visualization

---

# 🏗️ System Architecture

The complete system follows a **Frontend → REST API → ML Engine → Response** architecture.

```text
                         ┌─────────────────────┐
                         │       USER          │
                         │                     │
                         │ "I failed my test"  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    React Frontend   │
                         │      + Vite         │
                         │                     │
                         │  Text Input         │
                         │  Emotion UI         │
                         │  Result Visualizer  │
                         └──────────┬──────────┘
                                    │
                              HTTP POST
                           /api/analyze
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI         │
                         │      Backend        │
                         │                     │
                         │ Request Validation  │
                         │ CORS                │
                         │ API Routing          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       Emotion Engine         │
                    │                              │
                    │ Sentence Transformer         │
                    │        ↓                     │
                    │ Semantic Embedding           │
                    │        ↓                     │
                    │ Cosine Similarity            │
                    │        ↓                     │
                    │ Emotion Prototypes           │
                    └──────────────┬───────────────┘
                                   │
                                   │ Supporting ML
                                   ▼
                    ┌──────────────────────────────┐
                    │  Unsupervised ML Pipeline    │
                    │                              │
                    │ Deep Autoencoder             │
                    │        ↓                     │
                    │ Latent Representation        │
                    │        ↓                     │
                    │ UMAP                         │
                    │        ↓                     │
                    │ HDBSCAN                      │
                    │        ↓                     │
                    │ Cluster Discovery            │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   AnalyzeResponse   │
                         │                     │
                         │ Primary Emotion     │
                         │ Confidence          │
                         │ Top 5 Emotions      │
                         │ Cluster             │
                         │ Explanation         │
                         └──────────┬──────────┘
                                    │
                              JSON Response
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    React Frontend   │
                         │                     │
                         │ Emotional Map       │
                         │ Confidence Bars     │
                         │ Emotion Orb         │
                         └─────────────────────┘
```

---

# 🧬 Machine Learning Backbone

The ML pipeline can be visualized as:

```text
                 RAW TEXT
                    │
                    ▼
        ┌──────────────────────┐
        │ Sentence Transformer │
        │ all-MiniLM-L6-v2     │
        └──────────┬───────────┘
                   │
                   ▼
          Semantic Embedding
                   │
          ┌────────┴─────────┐
          │                  │
          ▼                  ▼
   Emotion Prototype     Unsupervised
      Matching             Discovery
          │                  │
          │                  ▼
          │          StandardScaler
          │                  │
          │                  ▼
          │          Deep Autoencoder
          │                  │
          │                  ▼
          │          Latent Space
          │                  │
          │                  ▼
          │                 UMAP
          │                  │
          │                  ▼
          │               HDBSCAN
          │                  │
          │                  ▼
          │             Cluster ID
          │
          ▼
   Cosine Similarity
          │
          ▼
   Emotion Scores
          │
          └────────────┬─────────────┐
                       │             │
                       ▼             ▼
                Primary Emotion   Top 5 Emotions
                       │             │
                       └──────┬──────┘
                              ▼
                       API JSON Response
```


# 🔬 How the Emotion Engine Works

## 1. Text Encoding

The user's sentence is passed to:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts the sentence into a numerical semantic representation.

```python
embedding = encoder.encode(
    [text],
    normalize_embeddings=True
)
```

---

## 2. Emotion Prototypes

The system defines semantic descriptions for 15 emotions.

For example:

```python
"sadness":
    "I feel sad, hurt, low, heartbroken,
     unhappy and emotionally down."
```

Each emotion description is converted into an embedding.

---

## 3. Semantic Similarity

The user's embedding is compared with every emotion prototype using:

```text
Cosine Similarity
```

Conceptually:

```text
User Sentence
      │
      ▼
Embedding
      │
      ├──────► Joy
      ├──────► Sadness
      ├──────► Anxiety
      ├──────► Guilt
      ├──────► Hope
      └──────► ...
```

The similarities are converted into readable scores.

---

## 4. Ranking

The emotions are sorted according to their semantic similarity.

The system returns the top five signals.

```text
Input
 ↓
15 emotion scores
 ↓
Sort descending
 ↓
Top 5
 ↓
Primary emotion = highest score
```

---

# 🧠 Unsupervised Learning Pipeline

SMH Cheerful also contains a separate unsupervised representation-learning component.

### Step 1 — Sentence Embeddings

The unlabeled corpus is encoded into embeddings.

```text
sample_corpus.txt
       ↓
Sentence Transformer
       ↓
Embedding Matrix
```

---

### Step 2 — Standardization

The embeddings are normalized using:

```python
StandardScaler()
```

---

### Step 3 — Deep Autoencoder

A PyTorch autoencoder learns a compressed representation.

```text
Input
 ↓
256 neurons
 ↓
128 neurons
 ↓
64-dimensional latent space
 ↓
128 neurons
 ↓
256 neurons
 ↓
Reconstructed Input
```

The encoder learns a compact representation of the input data.

---

### Step 4 — UMAP

UMAP reduces the learned latent representation while preserving meaningful structure.

```text
64D Latent Space
       ↓
      UMAP
       ↓
8D Representation
```

---

### Step 5 — HDBSCAN

HDBSCAN identifies naturally occurring dense groups within the reduced representation.

```text
UMAP Representation
        ↓
     HDBSCAN
        ↓
Cluster 0
Cluster 1
Cluster 2
Cluster 3
...
```

Unlike traditional K-Means, HDBSCAN does not require specifying the exact number of clusters beforehand.

---

# 🧩 Technology Stack

## Frontend

* **React 19**
* **Vite 6**
* JavaScript / JSX
* CSS3
* Responsive UI
* CSS animations

## Backend

* **Python 3.11**
* **FastAPI**
* **Uvicorn**
* **Pydantic**

## Machine Learning

* **PyTorch**
* **Sentence Transformers**
* **scikit-learn**
* **UMAP**
* **HDBSCAN**
* **NumPy**

## DevOps

* Docker
* Docker Compose
* GitHub Actions

---

# 📁 Project Structure

```text
SMH-Cheerful/
│
├── .github/
│   └── workflows/
│       └── docker-build.yml
│
├── backend/
│   │
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── emotion_engine.py
│   │   │
│   │   └── ml/
│   │       └── unsupervised.py
│   │
│   ├── data/
│   │   └── sample_corpus.txt
│   │
│   ├── scripts/
│   │   └── train_unsupervised.py
│   │
│   ├── artifacts/
│   │   └── .gitkeep
│   │
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   │
│   ├── src/
│   │   ├── components/
│   │   │   ├── Intro.jsx
│   │   │   ├── Analyzer.jsx
│   │   │   └── EmotionOrb.jsx
│   │   │
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   │
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
│
├── docker-compose.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

# 🚀 Getting Started

There are two ways to run SMH Cheerful:

1. 🐳 Docker — recommended
2. 💻 Run frontend and backend separately

---

# 🐳 Option 1 — Run with Docker

### Requirements

Install:

* Docker
* Docker Compose

Then clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SMH-Cheerful.git
cd SMH-Cheerful
```

Build and start the application:

```bash
docker compose up --build
```

Once the containers are running:

### Frontend

```text
http://localhost:5173
```

### Backend

```text
http://localhost:8000
```

### FastAPI Swagger Documentation

```text
http://localhost:8000/docs
```

To stop the application:

```bash
docker compose down
```

---

# 💻 Option 2 — Run Manually

## 1. Start the Backend

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

The API will run at:

```text
http://localhost:8000
```

---

## 2. Start the Frontend

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start Vite:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

---

# 🔌 API Documentation

SMH Cheerful exposes a REST API through FastAPI.

## Health Check

```http
GET /health
```

Example:

```json
{
  "status": "ok",
  "service": "smh-cheerful"
}
```

---

## Analyze Emotion

```http
POST /api/analyze
```

### Request

```json
{
  "text": "I failed my test and I feel terrible about disappointing everyone."
}
```

### Response

```json
{
  "primary_emotion": "sadness",
  "confidence": 0.82,
  "emotions": [
    {
      "emotion": "sadness",
      "score": 0.82
    },
    {
      "emotion": "guilt",
      "score": 0.76
    },
    {
      "emotion": "anxiety",
      "score": 0.59
    }
  ],
  "cluster": 3,
  "explanation": "The language is semantically closest to sadness, guilt and anxiety."
}
```

---

# 🔄 Request Flow

When the user presses **"Reveal emotions →"**:

```text
User enters text
       ↓
React captures input
       ↓
POST /api/analyze
       ↓
FastAPI validates request
       ↓
EmotionEngine.analyze()
       ↓
Sentence Transformer
       ↓
Semantic Embedding
       ↓
Cosine Similarity
       ↓
Emotion Ranking
       ↓
Optional HDBSCAN Cluster
       ↓
JSON Response
       ↓
React receives result
       ↓
Emotion Orb + Scores + Explanation
```

---

# ⚙️ Configuration

The backend supports changing the Sentence Transformer model through:

```text
MODEL_NAME
```

The default model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Docker Compose currently configures:

```yaml
environment:
  - MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

The frontend API URL can be configured through:

```text
VITE_API_URL
```

If it is not provided, the frontend defaults to:

```text
http://localhost:8000
```

---

# 🧪 Training the Unsupervised Pipeline

The repository includes:

```text
backend/scripts/train_unsupervised.py
```

From the backend directory:

```bash
python scripts/train_unsupervised.py
```

The unsupervised pipeline uses:

```text
Unlabeled Corpus
      ↓
Sentence Embeddings
      ↓
StandardScaler
      ↓
Deep Autoencoder
      ↓
Latent Representation
      ↓
UMAP
      ↓
HDBSCAN
      ↓
Cluster Discovery
```

For meaningful research experiments, the sample corpus can be replaced with a substantially larger and more diverse unlabeled dataset.

---

# 🔍 Important ML Design Decision

One important distinction in this project is:

> **Clustering discovers structure; semantic prototypes provide human-readable emotion alignment.**

An unsupervised clustering algorithm does not inherently know that a particular cluster represents:

```text
Sadness
```

or:

```text
Anxiety
```

HDBSCAN simply discovers groups of semantically similar representations.

SMH Cheerful therefore uses **semantic prototype alignment** to connect discovered language patterns with interpretable emotion concepts.

This makes the architecture easier to explain and prevents the project from incorrectly claiming that an unsupervised algorithm directly learned human emotion labels.

---

# 🛡️ Limitations

SMH Cheerful is a **linguistic emotion-analysis project**, not a clinical psychological system.

The scores represent semantic similarity and should not be interpreted as medically validated probabilities.

The system may also struggle with:

* Sarcasm
* Complex mixed emotions
* Cultural context
* Very short inputs
* Metaphorical language
* Ambiguous statements
* Unusual slang
* Context that exists outside the provided sentence

---

# 🔮 Future Improvements

Potential future versions could include:

### 🧠 Better Emotion Models

Compare the current semantic-prototype approach against:

* Fine-tuned transformer classifiers
* GoEmotions-based models
* RoBERTa emotion models
* DistilBERT
* Larger sentence embedding models

### 📚 Larger Dataset

Replace the small demonstration corpus with a larger, diverse unlabeled dataset.

### 🎯 Supervised vs Unsupervised Benchmark

Create an experiment comparing:

```text
Supervised Classification
          VS
Prototype Similarity
          VS
Unsupervised Clustering
```

using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Silhouette Score
* Cluster quality

### 🧩 Context-Aware Analysis

Extend the system to analyze multiple messages:

```text
Message 1
   ↓
Message 2
   ↓
Message 3
   ↓
Conversation-level emotional trajectory
```

### 📈 Emotion Timeline

Visualize how emotions change throughout a conversation.

### 🌐 Production Deployment

Possible architecture:

```text
React
  ↓
CDN
  ↓
API Gateway
  ↓
FastAPI
  ↓
ML Inference Service
  ↓
Model Server
```

---

# 🐳 Deployment Architecture

The repository is already containerized into two services:

```text
                 Docker Compose
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      ┌─────────────┐     ┌─────────────┐
      │  Frontend   │     │   Backend   │
      │             │     │             │
      │ React/Vite  │────►│  FastAPI    │
      │ Port 5173   │     │  Port 8000  │
      └─────────────┘     └──────┬──────┘
                                  │
                                  ▼
                         Sentence Transformer
                                  │
                                  ▼
                         Emotion Engine
                                  │
                                  ▼
                         ML Pipeline
```

This separation makes the project easier to develop, test, and deploy independently.

---

# 🔧 GitHub Actions

The project also includes a GitHub Actions workflow:

```text
.github/workflows/docker-build.yml
```

This allows Docker builds to be automatically checked through GitHub's CI workflow.

---

# 📌 Example Use Cases

SMH Cheerful can serve as a foundation for:

* 💬 AI chat applications
* 🧠 Emotion-aware interfaces
* 📊 Customer feedback analysis
* 📝 Journal analysis
* 🎓 Student wellbeing research
* 🤖 Conversational AI
* 🛍️ Customer experience systems
* 📱 Social applications
* 🔬 NLP research experiments

These applications would require additional validation and safeguards before being used in high-stakes environments.

---

# 📜 Disclaimer

SMH Cheerful estimates **linguistic emotion signals** from text.

It is **not** a medical or psychological diagnostic system.

Do not use the output for:

* Medical diagnosis
* Mental-health diagnosis
* Crisis assessment
* Hiring decisions
* Insurance decisions
* Academic disciplinary decisions
* Other high-stakes decisions

---

# 👩‍💻 Author

**Nainika Agrawal**

B.Tech — Electronics & Communication Engineering

Interested in:

```text
Artificial Intelligence
Machine Learning
Software Engineering
NLP
Cloud & Backend Systems
Research
```

---

# ⭐ If You Like This Project

If SMH Cheerful was useful or interesting:

⭐ Star the repository
🍴 Fork the project
🐛 Open an issue
💡 Suggest an improvement
🔀 Submit a pull request

---

## 🌸 SMH Cheerful

> **Because human emotions are rarely just one label.**

Built with **React · FastAPI · PyTorch · Sentence Transformers · UMAP · HDBSCAN**
