from __future__ import annotations

from typing import Any


# Hardcoded, in-memory registry. In a real system this would live in a DB.
AGENTS: list[dict[str, Any]] = [
    {
        "id": "agent_flight_travel",
        "name": "FlightTravelAgent",
        "endpoint": "http://localhost:9006/run",
        "metadata": {
            "description": "Finds and explains flight options from a local flights dataset; can chat with session memory.",
            "capabilities": ["flight_search", "travel_planning", "conversational_qa"],
            "tags": ["travel", "flights", "booking", "itinerary"],
        },
    },
    {
        "id": "agent_web_research",
        "name": "WebResearchAgent",
        "endpoint": "http://localhost:9001/run",
        "metadata": {
            "description": "Finds up-to-date info on the web, summarizes sources, and extracts key facts.",
            "capabilities": ["web_search", "summarization", "citation"],
            "tags": ["research", "web", "fact-checking"],
        },
    },
    {
        "id": "agent_code_helper",
        "name": "CodeHelperAgent",
        "endpoint": "http://localhost:9002/run",
        "metadata": {
            "description": "Helps implement features, debug errors, and explain code changes.",
            "capabilities": ["coding", "debugging", "refactoring"],
            "tags": ["engineering", "python", "typescript", "fastapi"],
        },
    },
    {
        "id": "agent_data_analyst",
        "name": "DataAnalystAgent",
        "endpoint": "http://localhost:9003/run",
        "metadata": {
            "description": "Analyzes datasets, writes SQL, and produces charts/insights.",
            "capabilities": ["analysis", "sql", "visualization"],
            "tags": ["data", "analytics", "sql"],
        },
    },
    {
        "id": "agent_product_writer",
        "name": "ProductWriterAgent",
        "endpoint": "http://localhost:9004/run",
        "metadata": {
            "description": "Writes product copy, docs, and user-friendly explanations.",
            "capabilities": ["writing", "documentation", "tone_control"],
            "tags": ["docs", "copywriting", "product"],
        },
    },
    {
        "id": "agent_support_triage",
        "name": "SupportTriageAgent",
        "endpoint": "http://localhost:9005/run",
        "metadata": {
            "description": "Classifies user issues, suggests next actions, and routes to the right team/agent.",
            "capabilities": ["classification", "triage", "routing"],
            "tags": ["support", "ops", "triage"],
        },
    },
]


def get_agent_by_id(agent_id: str) -> dict[str, Any] | None:
    for agent in AGENTS:
        if agent["id"] == agent_id:
            return agent
    return None

