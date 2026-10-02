from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .emotion_engine import EmotionEngine
from .schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(
    title="SMH Cheerful API",
    description="Unsupervised human emotion analysis API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = EmotionEngine()


@app.get("/health")
def health():
    return {"status": "ok", "service": "smh-cheerful"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return engine.analyze(text)
