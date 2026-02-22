from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Agent(BaseModel):
    id: str
    name: str
    endpoint: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    intent: str = Field(..., description="User intent / task description.")
    # This registry is intentionally simple: always return 3 best-fitting agents.
    top_k: int = Field(3, ge=3, le=3, description="How many agents to return (fixed at 3).")
    model: str = Field(
        "gpt-4.1-mini",
        description="OpenAI model name used to rank/select agents.",
    )


class SearchResult(BaseModel):
    rank: int = Field(..., ge=1, description="1 = best fit.")
    agent_id: str
    endpoint: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field(..., description="Why this agent is a good fit for the given intent.")
    trust: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence/trust in this recommendation (0 to 1).",
    )
