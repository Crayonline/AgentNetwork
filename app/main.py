from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.models import (
    Agent,
    SearchRequest,
    SearchResult,
)
from app.recommender import fallback_recommend, openai_recommend
from app.registry import AGENTS, get_agent_by_id

load_dotenv()

app = FastAPI(title="AgentGate Registry", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents", response_model=list[Agent])
def list_agents() -> list[Agent]:
    return [Agent(**a) for a in AGENTS]


@app.get("/agents/{agent_id}", response_model=Agent)
def get_agent(agent_id: str) -> Agent:
    agent = get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="agent not found")
    return Agent(**agent)


@app.post("/search", response_model=list[SearchResult])
def search(payload: SearchRequest) -> list[SearchResult]:
    """
    User-facing endpoint: given an intent string, returns the 3 best agents
    from the local hardcoded registry.
    """
    try:
        if os.getenv("OPENAI_API_KEY"):
            results = openai_recommend(payload.intent, payload.top_k, payload.model)
        else:
            results = fallback_recommend(payload.intent, payload.top_k)
    except Exception:
        results = fallback_recommend(payload.intent, payload.top_k)

    return [SearchResult(**r) for r in results]

