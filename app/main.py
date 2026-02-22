from __future__ import annotations

import importlib
import os
import pkgutil

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.models import Agent, SearchRequest, SearchResult
from app.core.recommender import fallback_recommend, openai_recommend
from app.core.registry import agent_registry

import app.agents as agents_pkg  # noqa: F401 - This is why agents/__init__.py must exist

load_dotenv()

app = FastAPI(title="AgentGate Network Hub", version="0.2.0")

# 2. Add this block right after 'app = FastAPI(...)'
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🔌 PLUGIN DISCOVERY ENGINE
# ==========================================
print("🔍 Scanning for Agent plugins...")
for _, module_name, _ in pkgutil.iter_modules(agents_pkg.__path__):
    module = importlib.import_module(f"app.agents.{module_name}")

    # 1. If the plugin has an APIRouter, mount its endpoints!
    if hasattr(module, "router"):
        app.include_router(module.router)

    # 2. If the plugin has a setup() function, run it to register metadata!
    if hasattr(module, "setup"):
        module.setup(agent_registry)

# ==========================================
# 🌐 AGENTHUB ROUTING ENDPOINTS
# ==========================================
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "agents_online": len(agent_registry.get_all())}


@app.get("/agents", response_model=list[Agent])
def list_agents():
    return [Agent(**a) for a in agent_registry.get_all()]


@app.post("/search", response_model=list[SearchResult])
def search(payload: SearchRequest):
    try:
        if os.getenv("OPENAI_API_KEY"):
            results = openai_recommend(payload.intent, payload.top_k, payload.model)
        else:
            results = fallback_recommend(payload.intent, payload.top_k)
    except Exception:
        results = fallback_recommend(payload.intent, payload.top_k)
    return [SearchResult(**r) for r in results]
