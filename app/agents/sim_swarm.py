from __future__ import annotations

import os
import random
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from app.core.registry import AgentRegistry, agent_registry

router = APIRouter()

NUM_SIMULATED_AGENTS = 50

# Just a small sample of domains!
DOMAINS = [
    ("E-Commerce", ["shopping", "retail", "discounts"], ["product_search", "checkout"]),
    ("Legal", ["legal", "contracts", "law"], ["contract_analysis", "compliance_check"]),
    ("Healthcare", ["health", "medical", "wellness"], ["symptom_triage", "doctor_booking"]),
    ("Travel", ["travel", "booking", "flight"], ["flight_search", "travel_planning"]),
    ("Restaurant", ["restaurant", "booking", "food"], ["restaurant_search", "restaurant_booking"]),
    ("Hotel", ["hotel", "booking", "accommodation"], ["hotel_search", "hotel_booking"]),
    ("Car Rental", ["car rental", "booking", "transport"], ["car_rental_search", "car_rental_booking"]),
    ("Event", ["event", "booking", "ticket"], ["event_search", "event_booking"]),
    ("Music", ["music", "concert", "ticket"], ["music_search", "music_booking"]),
    ("Movie", ["movie", "ticket"], ["movie_search", "movie_booking"]),
    ("Book", ["book", "purchase"], ["book_search", "book_purchase"]),
    ("Clothing", ["clothing", "purchase"], ["clothing_search", "clothing_purchase"]),
    ("Electronics", ["electronics", "purchase"], ["electronics_search", "electronics_purchase"]),
    ("Furniture", ["furniture", "purchase"], ["furniture_search", "furniture_purchase"]),
    ("Home", ["home", "purchase"], ["home_search", "home_purchase"]),
    ("Garden", ["garden", "purchase"], ["garden_search", "garden_purchase"]),
    ("Pet", ["pet", "purchase"], ["pet_search", "pet_purchase"]),
    ("Toy", ["toy", "purchase"], ["toy_search", "toy_purchase"]),
    ("Art", ["art", "purchase"], ["art_search", "art_purchase"]),
    ("Groceries", ["groceries", "purchase"], ["groceries_search", "groceries_purchase"]),
]

# Multi-tenant memory: agent_id -> session_id -> history
sim_memory: Dict[str, Dict[str, List[Dict]]] = {}


def setup(registry: AgentRegistry) -> None:
    """Dynamically generates and registers the swarm."""
    for _ in range(NUM_SIMULATED_AGENTS):
        domain_name, tags, caps = random.choice(DOMAINS)
        agent_id = f"sim_{uuid.uuid4().hex[:8]}"

        registry.register({
            "id": agent_id,
            "name": f"Smart{domain_name}Agent",
            "endpoint": f"http://localhost:8000/sim/{agent_id}/run",
            "metadata": {
                "description": f"Handles specialized requests for {domain_name}.",
                "capabilities": random.sample(caps, k=min(2, len(caps))),
                "tags": random.sample(tags, k=min(2, len(tags))),
            }
        })


class SimAgentRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


# We use a wildcard route so this one function powers ALL simulated agents!
@router.post("/sim/{agent_id}/run")
def run_simulated_agent(agent_id: str, req: SimAgentRequest):
    agent = agent_registry.get_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Setup Session Memory
    if agent_id not in sim_memory:
        sim_memory[agent_id] = {}

    if req.session_id is None or req.session_id not in sim_memory[agent_id]:
        session_id = str(uuid.uuid4())
        sim_memory[agent_id][session_id] = []
    else:
        session_id = req.session_id

    history = sim_memory[agent_id][session_id]
    history.append({"role": "user", "content": req.prompt})

    # Let the AI Roleplay as the randomized agent
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        answer = f"[{agent['name']}] I am online, but my OPENAI_API_KEY is missing so I cannot process tasks."
    else:
        client = OpenAI(api_key=api_key)

        sys_prompt = (
            f"You are {agent['name']}, a highly specialized AgentGate NetAgent.\n"
            f"Description: {agent['metadata']['description']}\n"
            f"Your capabilities: {', '.join(agent['metadata']['capabilities'])}\n"
            f"Your tags: {', '.join(agent['metadata']['tags'])}\n\n"
            "Rules:\n"
            "1. Stay completely in character as this specialized agent.\n"
            "2. If the user asks for something outside your domain, refuse and tell them to query the AgentHub for a different agent.\n"
            "3. Be concise, professional, and transactional."
        )

        messages = [{"role": "system", "content": sys_prompt}] + history

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        answer = response.choices[0].message.content

    history.append({"role": "assistant", "content": answer})

    return {
        "session_id": session_id,
        "answer": answer,
        "conversation_history": history
    }
