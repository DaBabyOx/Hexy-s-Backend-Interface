from __future__ import annotations

from pathlib import Path
import asyncio
import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .mujoco_sim import MujocoSimulator, MujocoState


class ErrorResponse(BaseModel):
    detail: str


def _parse_cors_origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _resolve_model_path() -> Path:
    env_model = os.getenv("MODEL_PATH")
    if env_model:
        return Path(env_model)

    rl_root = os.getenv("RL_ROOT")
    if rl_root:
        candidate = Path(rl_root) / "models" / "hexapod_static.xml"
        return candidate

    # Fallback: sibling folder when running from monorepo layout.
    return Path(__file__).resolve().parents[2] / "VendorAgnosticRL" / "models" / "hexapod_static.xml"


def _resolve_assets_root() -> Path:
    rl_root = os.getenv("RL_ROOT")
    if rl_root:
        return Path(rl_root)

    return Path(__file__).resolve().parents[2] / "VendorAgnosticRL"


_simulator: MujocoSimulator | None = None
_simulator_error: Exception | None = None


def _get_simulator() -> MujocoSimulator:
    global _simulator, _simulator_error
    if _simulator is not None:
        return _simulator
    if _simulator_error is not None:
        raise _simulator_error

    model_path = _resolve_model_path()
    if not model_path.exists():
        _simulator_error = FileNotFoundError(
            f"Model file not found at {model_path}. Set RL_ROOT or MODEL_PATH."
        )
        raise _simulator_error

    try:
        _simulator = MujocoSimulator(model_path)
    except Exception as exc:  # pragma: no cover - runtime dependency
        _simulator_error = exc
        raise
    return _simulator


app = FastAPI(title="Hexy Backend", version="0.1.0")

assets_root = _resolve_assets_root()
if assets_root.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_root)), name="assets")

cors_origins = _parse_cors_origins(
    os.getenv("CORS_ORIGINS", "http://localhost:5173")
)
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/mujoco/state", response_model=MujocoState, responses={503: {"model": ErrorResponse}})
def mujoco_state() -> MujocoState:
    try:
        simulator = _get_simulator()
        return simulator.state()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/mujoco/reset", response_model=MujocoState, responses={503: {"model": ErrorResponse}})
def mujoco_reset() -> MujocoState:
    try:
        simulator = _get_simulator()
        return simulator.reset()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.websocket("/mujoco/stream")
async def mujoco_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    raw_interval = websocket.query_params.get("interval_ms", "50")
    try:
        interval_ms = int(raw_interval)
    except ValueError:
        await websocket.send_json({"detail": "interval_ms must be an integer"})
        await websocket.close(code=1003)
        return

    interval_ms = max(10, min(interval_ms, 1000))

    try:
        simulator = _get_simulator()
    except Exception as exc:
        await websocket.send_json({"detail": str(exc)})
        await websocket.close(code=1011)
        return

    try:
        while True:
            state = simulator.state().model_dump()
            await websocket.send_json(state)
            await asyncio.sleep(interval_ms / 1000)
    except WebSocketDisconnect:
        return
