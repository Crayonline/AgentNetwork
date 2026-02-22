from __future__ import annotations

import os
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI

from app.core.registry import AgentRegistry

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore[assignment]

# 1. Expose an APIRouter
router = APIRouter()

# 2. Define the Setup function for auto-discovery
def setup(registry: AgentRegistry) -> None:
    registry.register({
        "id": "agent_flight_travel",
        "name": "FlightTravelAgent",
        "endpoint": "http://localhost:8000/flight/run",
        "metadata": {
            "description": "Finds and explains flight options from a local dataset.",
            "capabilities": ["flight_search", "travel_planning"],
            "tags": ["travel", "flights", "booking"],
        }
    })

# --- Types & Logic ---
class AgentRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    session_id: str
    answer: str
    conversation_history: List[Dict]


# Dataset state
_df = None
_dataset_load_error: Optional[str] = None
_available_cities: Optional[list[str]] = None
_client: Optional[OpenAI] = None
memory: Dict[str, List[Dict]] = {}


def _load_dataset() -> None:
    global _df, _dataset_load_error, _available_cities
    if _df is not None:
        return

    if pd is None:
        _dataset_load_error = "pandas is required. Install pandas and retry."
        raise RuntimeError(_dataset_load_error)

    csv_path = os.getenv("FLIGHTS_CSV_PATH") or os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "airlines_flights_data.csv",
    )

    if not os.path.exists(csv_path):
        _dataset_load_error = (
            "Flights dataset CSV not found. "
            "Set FLIGHTS_CSV_PATH or ensure app/airlines_flights_data.csv exists."
        )
        raise RuntimeError(_dataset_load_error)

    _df = pd.read_csv(csv_path)
    try:
        src = _df["source_city"].dropna().astype(str).unique().tolist()
        dst = _df["destination_city"].dropna().astype(str).unique().tolist()
        _available_cities = sorted(set(src) | set(dst), key=lambda x: x.lower())
    except Exception:
        _available_cities = None


def _search_flights(prompt: str):
    if _df is None:
        _load_dataset()

    prompt_lower = prompt.lower()
    cities = _available_cities or []
    mentioned = [c for c in cities if c.lower() in prompt_lower]
    mentioned.sort(key=lambda c: prompt_lower.find(c.lower()))

    results = _df

    if len(mentioned) >= 2:
        src_city, dst_city = mentioned[0], mentioned[1]
        results = results[
            (results["source_city"].astype(str).str.lower() == src_city.lower())
            & (results["destination_city"].astype(str).str.lower() == dst_city.lower())
        ]
    elif len(mentioned) == 1:
        city = mentioned[0]
        results = results[
            (results["source_city"].astype(str).str.lower() == city.lower())
            | (results["destination_city"].astype(str).str.lower() == city.lower())
        ]

    try:
        top = results.nsmallest(5, "price")
    except Exception:
        top = results.head(5)

    if top is None or top.empty:
        try:
            top = _df.nsmallest(5, "price")
        except Exception:
            top = _df.head(5)

    return top


def _generate_answer(prompt: str, session_history: List[Dict]) -> str:
    full_context = ""
    for msg in session_history:
        if msg["role"] == "user":
            full_context += msg["content"] + " "
    full_context += prompt

    flights = _search_flights(full_context)

    if flights.empty:
        cities_hint = ""
        if _available_cities:
            cities_hint = " Available cities include: " + ", ".join(_available_cities[:12]) + "."
        dataset_context = "No matching flights found in dataset." + cities_hint
    else:
        dataset_context = ""
        for _, row in flights.iterrows():
            dataset_context += (
                f"{row['airline']} flight {row['flight']} "
                f"from {row['source_city']} to {row['destination_city']}, "
                f"departure {row['departure_time']}, "
                f"arrival {row['arrival_time']}, "
                f"duration {row['duration']} hours, "
                f"class {row['class']}.\n"
            )

    if _client is None:
        return (
            "I can search the flights dataset, but I can't generate a full chat response because OPENAI_API_KEY "
            "is not set.\n\n"
            f"Top matches:\n{dataset_context}".strip()
        )

    messages = [
        {
            "role": "system",
            "content": f"""
You are a flight assistant agent in the AgentGate network.

Use the following dataset information to answer the user.

Dataset flights:
{dataset_context}

Be conversational, helpful, and concise like ChatGPT.
"""
        }
    ]
    for msg in session_history:
        messages.append(msg)
    messages.append({"role": "user", "content": prompt})

    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "I found flight options but could not generate a chat response (API unavailable).\n\n"
            f"Top matches:\n{dataset_context}".strip()
        )


# Lazy init OpenAI client on first request
def _ensure_client() -> None:
    global _client
    if _client is None and os.getenv("OPENAI_API_KEY"):
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 3. Mount your specific endpoints to the router
@router.post("/flight/run", response_model=AgentResponse)
def run_flight(req: AgentRequest) -> AgentResponse:
    _ensure_client()

    if req.session_id is None or req.session_id not in memory:
        session_id = str(uuid.uuid4())
        memory[session_id] = []
    else:
        session_id = req.session_id

    memory[session_id].append({"role": "user", "content": req.prompt})
    answer = _generate_answer(req.prompt, memory[session_id])
    memory[session_id].append({"role": "assistant", "content": answer})

    return AgentResponse(
        session_id=session_id,
        answer=answer,
        conversation_history=memory[session_id]
    )
