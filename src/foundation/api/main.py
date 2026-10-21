from fastapi import FastAPI
from pydantic import BaseModel

from foundation.api.schemas import HealthResponse
from foundation.rag.store import ask as rag_ask

app = FastAPI(title="AI Eng Foundation", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="ai-eng-foundation")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AI Eng Foundation API"}


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(payload: AskRequest):
    answer = rag_ask(payload.question)
    return {"answer": answer}
