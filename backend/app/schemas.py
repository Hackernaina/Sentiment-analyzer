from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=2000)


class EmotionScore(BaseModel):
    emotion: str
    score: float


class AnalyzeResponse(BaseModel):
    primary_emotion: str
    confidence: float
    emotions: list[EmotionScore]
    cluster: int | None
    explanation: str
