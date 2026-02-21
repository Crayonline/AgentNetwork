# AgentsNetwork

## Local setup

1) Create a local `.env` file (do **not** commit it). You can start from `env.example`.

2) Run the registry (router / agent search):

```bash
python run.py registry --reload --port 8000
```

3) Run the FlightTravel agent:

```bash
python run.py flighttravel --reload --port 9006
```

### Alternative (direct uvicorn)

```bash
uvicorn app.main:app --reload --port 8000
uvicorn app.FlightTravel:app --reload --port 9006
```

## Endpoints

- Registry:
  - `POST /search` body: `{"intent": "..."}`
- FlightTravel agent:
  - `POST /FlightTravel` body: `{"prompt":"...", "session_id": null}`
  - `GET /health`