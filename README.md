## AgentNetwork

### Run the Hub (single server)

All agents run on one server via the drop-in plugin architecture:

```bash
uvicorn app.main:app --reload --port 8000
```

Drop new agent files into `app/agents/` and restart to auto-register them.

### Endpoints

- **AgentHub**:
  - `GET /health` – status and agent count
  - `GET /agents` – list all registered agents
  - `POST /search` – body: `{"intent":"...", "top_k": 3}`
- **Flight agent**:
  - `POST /flight/run` – body: `{"prompt":"...", "session_id": null}`
- **Simulated agents**:
  - `POST /sim/{agent_id}/run` – body: `{"prompt":"...", "session_id": null}``
