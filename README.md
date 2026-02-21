## AgentNetwork

### Run services (separately)

- **Registry / agent search**:

```bash
uvicorn app.main:app --reload --port 8000
```

- **FlightTravel agent**:

```bash
uvicorn app.FlightTravel:app --reload --port 9006
```

### Run both at once (background)

This will start both servers in the background:

```bash
uvicorn app.FlightTravel:app --reload --port 9006 & uvicorn app.main:app --reload --port 8000 &
```

### Endpoints

- **Registry**:
  - `GET /health`
  - `GET /agents`
  - `POST /search` body: `{"intent":"..."}`
- **FlightTravel agent**:
  - `GET /health`
  - `POST /run` body: `{"prompt":"...", "session_id": null}`
