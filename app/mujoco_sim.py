from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import List

import numpy as np
from pydantic import BaseModel

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - handled by caller
    raise RuntimeError(
        "MuJoCo is not installed. Install the `mujoco` Python package first."
    ) from exc


class MujocoState(BaseModel):
    time: float
    qpos: List[float]
    qvel: List[float]
    ctrl: List[float]
    ncon: int
    nu: int
    njnt: int
    updated_at: float


class MujocoSimulator:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._lock = threading.Lock()

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> MujocoState:
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            return self._state_locked()

    def step(self, ctrl: List[float] | None, n_steps: int = 1) -> MujocoState:
        with self._lock:
            if ctrl is not None:
                if len(ctrl) != self.model.nu:
                    raise ValueError(
                        f"Expected {self.model.nu} control values, got {len(ctrl)}"
                    )
                self.data.ctrl[:] = np.array(ctrl, dtype=np.float64)
            else:
                self.data.ctrl[:] = 0.0

            for _ in range(max(1, n_steps)):
                mujoco.mj_step(self.model, self.data)
            return self._state_locked()

    def state(self) -> MujocoState:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> MujocoState:
        return MujocoState(
            time=float(self.data.time),
            qpos=[float(value) for value in self.data.qpos],
            qvel=[float(value) for value in self.data.qvel],
            ctrl=[float(value) for value in self.data.ctrl],
            ncon=int(self.data.ncon),
            nu=int(self.model.nu),
            njnt=int(self.model.njnt),
            updated_at=time.time(),
        )
