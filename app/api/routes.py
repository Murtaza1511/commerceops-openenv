from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException

from app.baseline_runner import run_all_tasks_local
from app.env.environment import CustomerSupportEnv
from app.env.grader import grade_task
from app.env.tasks import TASKS
from app.models.schemas import Action

router = APIRouter()
env = CustomerSupportEnv()


@router.get("/")
def index():
    return {
        "status": "ok",
        "service": "commerceops-openenv",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "commerceops-openenv",
    }


def _extract_task_id(payload: Any) -> Optional[str]:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("task_id", "id"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return None


@router.post("/reset")
def reset(payload: Any = Body(None)):
    task_id = _extract_task_id(payload) or TASKS[0]["id"]
    task = next((item for item in TASKS if item["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Invalid task_id")
    return env.reset_with_task(task)


@router.post("/step")
def step(action: Action):
    try:
        observation, reward, done, info = env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@router.get("/state")
def get_state():
    state = env.state()
    if state is None:
        raise HTTPException(status_code=404, detail="Environment has not been reset yet")
    return state


@router.get("/tasks")
def get_tasks():
    return {
        "tasks": TASKS,
        "action_schema": Action.model_json_schema(),
    }


@router.post("/grader")
def grader(task_id: Optional[str] = None, payload: Any = Body(None)):
    task_id = task_id or _extract_task_id(payload)
    if task_id is None:
        raise HTTPException(status_code=400, detail="task_id is required")

    task = next((item for item in TASKS if item["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Invalid task_id")

    state = env.state()
    if state is None:
        raise HTTPException(status_code=400, detail="Environment has not been reset yet")

    return {
        "task_id": task_id,
        "score": grade_task(state, task),
    }


@router.get("/baseline")
def run_baseline():
    return run_all_tasks_local()
