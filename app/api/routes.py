from typing import Any, Optional

from fastapi import APIRouter, Body

from app.baseline_runner import run_all_tasks_local
from app.env.environment import ApiRepairEnv
from app.env.grader import grade_task
from app.env.tasks import TASKS
from app.models.schemas import Action
from app.scoring import MIN_OPEN_SCORE, clamp_open_unit_interval, sanitize_score_fields

router = APIRouter()
env = ApiRepairEnv()
env.reset_with_task(TASKS[0])


@router.get("/")
def index():
    return {
        "status": "ok",
        "service": "apidebug-openenv",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "apidebug-openenv",
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


def _task_or_default(task_id: Optional[str]) -> dict:
    if isinstance(task_id, str):
        task = next((item for item in TASKS if item["id"] == task_id), None)
        if task is not None:
            return task
    return TASKS[0]


@router.post("/reset")
def reset(payload: Any = Body(None)):
    task_id = _extract_task_id(payload)
    task = _task_or_default(task_id)
    return env.reset_with_task(task)


@router.post("/step")
def step(action: Action):
    if env.state() is None or env.current_task is None:
        env.reset_with_task(TASKS[0])

    observation, reward, done, info = env.step(action)

    return sanitize_score_fields(
        {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
        }
    )


@router.get("/state")
def get_state():
    state = env.state()
    if state is None:
        env.reset_with_task(TASKS[0])
        state = env.state()
    return sanitize_score_fields(state.model_dump())


@router.get("/tasks")
def get_tasks():
    return sanitize_score_fields(
        {
        "tasks": TASKS,
        "action_schema": Action.model_json_schema(),
        }
    )


def _grade_response(task_id: Optional[str], payload: Any = None) -> dict:
    state = env.state()
    requested_task_id = task_id or _extract_task_id(payload)
    fallback_task_id = state.task_id if state is not None else None
    task = _task_or_default(requested_task_id or fallback_task_id)
    task_id = task["id"]

    if state is None:
        return sanitize_score_fields(
            {
            "task_id": task_id,
            "score": MIN_OPEN_SCORE,
            }
        )

    return sanitize_score_fields(
        {
        "task_id": task_id,
        "score": clamp_open_unit_interval(grade_task(state, task)),
        }
    )


@router.post("/grader")
def grader(task_id: Optional[str] = None, payload: Any = Body(None)):
    return _grade_response(task_id=task_id, payload=payload)


@router.get("/grader")
def grader_get(task_id: Optional[str] = None):
    return _grade_response(task_id=task_id)


@router.get("/baseline")
def run_baseline():
    return sanitize_score_fields(run_all_tasks_local())
