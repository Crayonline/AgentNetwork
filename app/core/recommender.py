from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from app.core.registry import agent_registry


def _normalize_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def fallback_recommend(intent: str, top_k: int = 3) -> list[dict[str, Any]]:
    """
    Lightweight local heuristic if OpenAI isn't available.
    Scores agents by token overlap with name/description/tags/capabilities.
    """
    agents = agent_registry.get_all()
    intent_tokens = set(_normalize_tokens(intent))
    scored: list[tuple[float, dict[str, Any], set[str], float]] = []

    for agent in agents:
        meta = agent.get("metadata", {}) or {}
        haystack = " ".join(
            [
                agent.get("name", ""),
                agent.get("id", ""),
                str(meta.get("description", "")),
                " ".join(meta.get("tags", []) or []),
                " ".join(meta.get("capabilities", []) or []),
            ]
        )
        hay_tokens = set(_normalize_tokens(haystack))
        overlap_tokens = intent_tokens & hay_tokens
        overlap = len(overlap_tokens)

        # Small bonus for explicit keyword hits.
        bonus = 0.0
        if "sql" in intent_tokens and "sql" in hay_tokens:
            bonus += 1.5
        if "debug" in intent_tokens and ("debugging" in hay_tokens or "debug" in hay_tokens):
            bonus += 1.0
        if "docs" in intent_tokens and ("documentation" in hay_tokens or "docs" in hay_tokens):
            bonus += 1.0

        score = overlap + bonus
        scored.append((score, agent, overlap_tokens, bonus))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[:top_k]

    results: list[dict[str, Any]] = []
    for i, (score, agent, overlap_tokens, bonus) in enumerate(chosen, start=1):
        meta = agent.get("metadata", {}) or {}
        caps = meta.get("capabilities", []) or []
        tags = meta.get("tags", []) or []

        # Map heuristic score -> 0..1 trust. This is intentionally simple and monotonic.
        # - more overlap => higher trust
        # - small bump for keyword bonus
        max_reasonable = max(6.0, float(len(intent_tokens) + 2))
        trust = min(1.0, max(0.05, score / max_reasonable))

        matched = sorted(overlap_tokens)[:8]
        reasoning_parts: list[str] = []
        if matched:
            reasoning_parts.append(f"Matched intent keywords: {', '.join(matched)}.")
        if bonus > 0:
            reasoning_parts.append("Extra boost from explicit capability keyword match.")
        if caps:
            reasoning_parts.append(f"Capabilities: {', '.join(caps[:6])}.")
        if tags:
            reasoning_parts.append(f"Tags: {', '.join(tags[:8])}.")
        if not reasoning_parts:
            reasoning_parts.append("Selected by default registry ranking (low-signal intent match).")

        reasoning = " ".join(reasoning_parts)

        results.append(
            {
                "rank": i,
                "agent_id": agent["id"],
                "endpoint": agent["endpoint"],
                "metadata": agent.get("metadata", {}) or {},
                "reasoning": reasoning,
                "trust": round(float(trust), 3),
            }
        )
    return results


def openai_recommend(intent: str, top_k: int, model: str) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    agents = agent_registry.get_all()

    # We ask the model to ONLY pick agent_ids and a rank order. We enrich afterwards.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "select_agents",
                "description": "Select the best matching agents for an intent from the provided registry.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "selections": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "rank": {"type": "integer", "minimum": 1},
                                    "agent_id": {"type": "string"},
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Short justification tailored to the user's intent.",
                                    },
                                    "trust": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Confidence score from 0 to 1.",
                                    },
                                },
                                "required": ["rank", "agent_id", "reasoning", "trust"],
                            },
                        }
                    },
                    "required": ["selections"],
                },
            },
        }
    ]

    registry_min = [
        {
            "id": a["id"],
            "name": a["name"],
            "endpoint": a["endpoint"],
            "metadata": a.get("metadata", {}) or {},
        }
        for a in agents
    ]

    system = (
        "You are an agent router. Choose the best agents from the registry for the user's intent. "
        "Return ONLY tool arguments; do not include any other text."
    )
    user = {
        "intent": intent,
        "top_k": top_k,
        "registry": registry_min,
        "rules": [
            "Pick distinct agent_id values from registry only.",
            "Use rank=1 for best fit, increasing for worse fits.",
            "Return exactly top_k selections if possible (else return as many as make sense).",
            "For each selection, provide a short reasoning (1-2 sentences) tied to intent and agent metadata.",
            "For each selection, provide trust in [0,1].",
        ],
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "select_agents"}},
    )

    msg = resp.choices[0].message
    if not msg.tool_calls:
        raise RuntimeError("OpenAI response did not include a tool call")

    raw_args = msg.tool_calls[0].function.arguments
    parsed = json.loads(raw_args)
    selections = parsed.get("selections", [])

    # Normalize, validate, enrich.
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for item in selections:
        agent_id = str(item.get("agent_id", "")).strip()
        if not agent_id or agent_id in seen:
            continue
        agent = agent_registry.get_by_id(agent_id)
        if not agent:
            continue
        seen.add(agent_id)
        reasoning = str(item.get("reasoning", "")).strip()
        trust_raw = item.get("trust", None)
        try:
            trust = float(trust_raw)
        except Exception:
            trust = 0.5
        trust = max(0.0, min(1.0, trust))
        if not reasoning:
            reasoning = "Selected by router based on best match between intent and agent metadata."

        normalized.append(
            {
                "rank": int(item.get("rank", len(normalized) + 1)),
                "agent_id": agent_id,
                "endpoint": agent["endpoint"],
                "metadata": agent.get("metadata", {}) or {},
                "reasoning": reasoning,
                "trust": round(float(trust), 3),
            }
        )

    # Ensure ranks are 1..N in returned order.
    normalized.sort(key=lambda x: x["rank"])
    normalized = normalized[:top_k]
    for i, item in enumerate(normalized, start=1):
        item["rank"] = i

    if not normalized:
        raise RuntimeError("OpenAI returned no valid selections")

    return normalized
