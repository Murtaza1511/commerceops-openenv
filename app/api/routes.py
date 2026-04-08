from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException

from app.baseline_runner import run_all_tasks_local
from app.env.environment import CustomerSupportEnv
from app.env.grader import grade_task
from app.env.tasks import TASKS
from app.models.schemas import Action
from app.scoring import MIN_OPEN_SCORE, clamp_open_unit_interval

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
        for key in ("task_id", "id", "task", "taskId"):
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


def _grade_response(task_id: Optional[str], payload: Any = None) -> dict:
    state = env.state()
    requested_task_id = task_id or _extract_task_id(payload)
    task_id = requested_task_id or (state.task_id if state is not None else TASKS[0]["id"])
    task = next((item for item in TASKS if item["id"] == task_id), None)
    if task is None:
        # Compatibility fallback for external validators that may pass
        # alternate task identifiers or omit task context.
        fallback_task_id = state.task_id if state is not None else TASKS[0]["id"]
        task = next((item for item in TASKS if item["id"] == fallback_task_id), None)
        if task is None:
            raise HTTPException(status_code=404, detail="Invalid task_id")
        task_id = task["id"]

    if state is None:
        return {
            "task_id": task_id,
            "score": MIN_OPEN_SCORE,
        }

    return {
        "task_id": task_id,
        "score": clamp_open_unit_interval(grade_task(state, task)),
    }


@router.post("/grader")
def grader(task_id: Optional[str] = None, payload: Any = Body(None)):
    return _grade_response(task_id=task_id, payload=payload)


@router.get("/grader")
def grader_get(task_id: Optional[str] = None):
    return _grade_response(task_id=task_id)


@router.get("/baseline")
def run_baseline():
    return run_all_tasks_local()
