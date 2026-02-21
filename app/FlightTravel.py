from __future__ import annotations

import os
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import kagglehub
except Exception as e:  # pragma: no cover
    kagglehub = None  # type: ignore[assignment]
from openai import OpenAI

load_dotenv()

# =========================
# Init OpenAI client
# =========================

client: Optional[OpenAI] = None

# =========================
# Load dataset
# =========================

df = None
dataset_load_error: Optional[str] = None


def _load_dataset() -> None:
    """
    Loads the flights dataset either from:
    - FLIGHTS_CSV_PATH (preferred, local path), or
    - KaggleHub download (requires kagglehub + network).
    """
    global df, dataset_load_error
    if df is not None:
        return

    if pd is None:
        dataset_load_error = "pandas is required to run FlightTravel agent. Install pandas and retry."
        raise RuntimeError(dataset_load_error)

    csv_path = os.getenv("FLIGHTS_CSV_PATH")
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return

    if kagglehub is None:
        dataset_load_error = (
            "Dataset not available. Set FLIGHTS_CSV_PATH to a local CSV, or install kagglehub to auto-download."
        )
        raise RuntimeError(dataset_load_error)

    try:
        path = kagglehub.dataset_download("rohitgrewal/airlines-flights-data")
        csv_file = None
        for file in os.listdir(path):
            if file.endswith(".csv"):
                csv_file = os.path.join(path, file)
                break

        if not csv_file:
            dataset_load_error = "KaggleHub download succeeded but no CSV was found in the dataset directory."
            raise RuntimeError(dataset_load_error)

        df = pd.read_csv(csv_file)
    except Exception as e:  # pragma: no cover
        dataset_load_error = f"Failed to load flights dataset: {e}"
        raise


def _ensure_dataset_loaded() -> Optional[str]:
    """
    Best-effort dataset loader.

    Returns:
        None if dataset is loaded, else an error message.
    """
    global dataset_load_error
    if df is not None:
        return None
    try:
        _load_dataset()
        return None
    except Exception as e:
        # _load_dataset should set dataset_load_error, but ensure we always return something user-friendly.
        if not dataset_load_error:
            dataset_load_error = str(e)
        return dataset_load_error

# =========================
# Conversation memory
# =========================

memory: Dict[str, List[Dict]] = {}

# =========================
# FastAPI
# =========================

app = FastAPI(title="AgentGate Flight Agent (ChatGPT-powered)")


@app.on_event("startup")
def _startup() -> None:
    global client
    # OpenAI client reads OPENAI_API_KEY from env; we don't hardcode keys.
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        client = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "dataset_loaded": df is not None,
        "dataset_error": dataset_load_error,
    }


class AgentRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    session_id: str
    answer: str
    conversation_history: List[Dict]


# =========================
# Search flights
# =========================


def search_flights(prompt):
    err = _ensure_dataset_loaded()
    if err:
        return None, err

    prompt_lower = prompt.lower()

    results = df.copy()

    # match source city
    results = results[
        results["source_city"].str.lower().apply(
            lambda x: any(city in prompt_lower for city in [x])
        )
    ]

    # match destination city
    results = results[
        results["destination_city"].str.lower().apply(
            lambda x: any(city in prompt_lower for city in [x])
        )
    ]

    return results.head(5), None

# =========================
# Generate ChatGPT answer
# =========================

def generate_answer(prompt, session_history):
    # Combine full conversation to preserve context
    full_context = ""

    for msg in session_history:
        if msg["role"] == "user":
            full_context += msg["content"] + " "

    full_context += prompt

    # Now search using full conversation context
    flights, dataset_err = search_flights(full_context)

    if dataset_err:
        dataset_context = f"Flights dataset unavailable: {dataset_err}"
    elif flights is None or flights.empty:
        dataset_context = "No matching flights found in dataset."
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

    # If OpenAI isn't configured, still return a useful deterministic response.
    if client is None:
        return (
            "I can search the flights dataset, but I can't generate a full chat response because OPENAI_API_KEY "
            "is not set.\n\n"
            f"Top matches:\n{dataset_context}".strip()
        )

    # Build conversation context
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

    # Add conversation history
    for msg in session_history:
        messages.append(msg)

    # Add new user message
    messages.append({"role": "user", "content": prompt})

    # Call ChatGPT API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # fast and cheap, perfect hackathon
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content


# =========================
# HTTP endpoint
# =========================

@app.post("/FlightTravel", response_model=AgentResponse)
def chat(req: AgentRequest):

    # Create or reuse session
    if req.session_id is None or req.session_id not in memory:
        session_id = str(uuid.uuid4())
        memory[session_id] = []
    else:
        session_id = req.session_id

    # Save user message
    memory[session_id].append({
        "role": "user",
        "content": req.prompt
    })

    # Generate ChatGPT answer
    answer = generate_answer(req.prompt, memory[session_id])

    # Save assistant answer
    memory[session_id].append({
        "role": "assistant",
        "content": answer
    })

    return AgentResponse(
        session_id=session_id,
        answer=answer,
        conversation_history=memory[session_id]
    )
