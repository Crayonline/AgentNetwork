from __future__ import annotations

from typing import Any, Dict, List


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, dict[str, Any]] = {}

    def register(self, agent_data: dict[str, Any]) -> None:
        """Dynamically register a new agent on startup."""
        self._agents[agent_data["id"]] = agent_data
        print(f"🔌 Auto-Registered: {agent_data['name']} ({agent_data['id']})")

    def get_all(self) -> List[dict[str, Any]]:
        return list(self._agents.values())

    def get_by_id(self, agent_id: str) -> dict[str, Any] | None:
        return self._agents.get(agent_id)


# The Singleton Instance
agent_registry = AgentRegistry()
