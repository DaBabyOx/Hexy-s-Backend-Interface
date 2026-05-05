Hexy backend (FastAPI) bridging the frontend to the MuJoCo simulation.

## Local run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set the MuJoCo model location:

```bash
set RL_ROOT=D:\Research\VendorAgnosticRL
```

4. Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

1. Copy .env.example to .env and set the host path for VendorAgnosticRL.
2. Build and run:

```bash
docker compose up --build
```

The API will be available at http://localhost:8000.

## API overview

- GET /health
- GET /mujoco/state
- POST /mujoco/reset
- WS /mujoco/stream?interval_ms=50
- Static assets: /assets/models/hexapod_static.xml (and /assets/STLFILES/*)
