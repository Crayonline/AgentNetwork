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
# NOTE: KaggleHub support is intentionally disabled for this project right now.
# We keep the code here (commented) in case you want to re-enable auto-download later.
#
# try:
#     import kagglehub
# except Exception as e:  # pragma: no cover
#     kagglehub = None  # type: ignore[assignment]
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
_available_cities: Optional[list[str]] = None


def _load_dataset() -> None:
    """
    Loads the flights dataset from a local CSV file.

    Priority:
    - FLIGHTS_CSV_PATH (if set), otherwise
    - app/airlines_flights_data.csv (checked in to this repo)
    """
    global df, dataset_load_error
    if df is not None:
        return

    if pd is None:
        dataset_load_error = "pandas is required to run FlightTravel agent. Install pandas and retry."
        raise RuntimeError(dataset_load_error)

    csv_path = os.getenv("FLIGHTS_CSV_PATH") or os.path.join(
        os.path.dirname(__file__),
        "airlines_flights_data.csv",
    )

    if not os.path.exists(csv_path):
        dataset_load_error = (
            "Flights dataset CSV not found. "
            "Set FLIGHTS_CSV_PATH or ensure app/airlines_flights_data.csv exists."
        )
        raise RuntimeError(dataset_load_error)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:  # pragma: no cover
        dataset_load_error = f"Failed to load flights dataset from {csv_path}: {e}"
        raise

    # Build a small index of cities for intent matching.
    global _available_cities
    try:
        src = df["source_city"].dropna().astype(str).unique().tolist()
        dst = df["destination_city"].dropna().astype(str).unique().tolist()
        _available_cities = sorted(set(src) | set(dst), key=lambda x: x.lower())
    except Exception:  # pragma: no cover
        _available_cities = None

    # ------------------------------------------------------------
    # KaggleHub auto-download (DISABLED / COMMENTED OUT)
    # ------------------------------------------------------------
    # If you ever want to re-enable this, a reasonable policy is:
    # - prefer FLIGHTS_CSV_PATH
    # - else try local app/airlines_flights_data.csv
    # - else fall back to KaggleHub download
    #
    # if kagglehub is None:
    #     dataset_load_error = (
    #         "Dataset not available. Set FLIGHTS_CSV_PATH to a local CSV, "
    #         "or install kagglehub to auto-download."
    #     )
    #     raise RuntimeError(dataset_load_error)
    #
    # try:
    #     path = kagglehub.dataset_download("rohitgrewal/airlines-flights-data")
    #     csv_file = None
    #     for file in os.listdir(path):
    #         if file.endswith(".csv"):
    #             csv_file = os.path.join(path, file)
    #             break
    #
    #     if not csv_file:
    #         dataset_load_error = "KaggleHub download succeeded but no CSV was found in the dataset directory."
    #         raise RuntimeError(dataset_load_error)
    #
    #     df = pd.read_csv(csv_file)
    # except Exception as e:  # pragma: no cover
    #     dataset_load_error = f"Failed to load flights dataset: {e}"
    #     raise

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
    if df is None:
        _load_dataset()

    prompt_lower = prompt.lower()

    # If the user doesn't mention cities, the old logic returned empty results.
    # Instead, we try to detect mentioned cities and fall back to "cheapest flights".
    cities = _available_cities or []
    mentioned = [c for c in cities if c.lower() in prompt_lower]
    mentioned.sort(key=lambda c: prompt_lower.find(c.lower()))

    results = df

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
    else:
        # No city mentioned -> just show cheap flights, so the user sees dataset-backed output.
        results = results

    # Prefer cheapest options if possible.
    try:
        top = results.nsmallest(5, "price")
    except Exception:
        top = results.head(5)

    # If route filter produced nothing, fall back to cheapest overall.
    if top is None or top.empty:
        try:
            top = df.nsmallest(5, "price")
        except Exception:
            top = df.head(5)

    return top

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
    flights = search_flights(full_context)

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

@app.post("/run", response_model=AgentResponse)
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
