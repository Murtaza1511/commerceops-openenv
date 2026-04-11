import json
import os
import warnings
from typing import Any, Dict

warnings.filterwarnings("ignore")

import requests

from app.baseline_runner import choose_action
from app.models.schemas import Action, Observation
from app.scoring import average_open_scores, clamp_open_unit_interval


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_env_with_default(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _log_start(task: Dict) -> None:
    print(f"[START] task={task['id']} difficulty={task['difficulty']}", flush=True)


def _log_step(task_id: str, step_index: int, reward: float) -> None:
    normalized_reward = clamp_open_unit_interval(float(reward))
    print(f"[STEP] task={task_id} step={step_index} reward={normalized_reward}", flush=True)


def _log_end(result: Dict) -> None:
    score_value = clamp_open_unit_interval(float(result["score"]))
    print(f"[END] task={result['task_id']} score={score_value}", flush=True)


def _log_results(results: list[Dict]) -> None:
    task_scores = []
    task_ids = []
    per_task = []
    for item in results:
        score_value = clamp_open_unit_interval(float(item["score"]))
        task_scores.append(score_value)
        task_ids.append(item["task_id"])
        per_task.append(
            {
                "task_id": item["task_id"],
                "difficulty": item["difficulty"],
                "task_score": score_value,
                "score": score_value,
            }
        )

    payload = {
        "task_scores": task_scores,
        "task_ids": task_ids,
        "per_task_scores": per_task,
        "average_task_score": average_open_scores(task_scores),
    }
    print(f"[RESULTS] {json.dumps(payload, sort_keys=False)}", flush=True)


def _extract_json_object(text: str) -> Dict:
    text = text.strip()
    if not text:
        raise RuntimeError("LLM returned an empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"LLM did not return valid JSON: {text}") from None
        return json.loads(text[start : end + 1])


def _fetch_tasks(env_base_url: str) -> list[Dict]:
    payload = requests.get(f"{env_base_url}/tasks", timeout=30).json()
    if not isinstance(payload, dict) or "tasks" not in payload:
        raise RuntimeError("Environment /tasks response is missing the tasks list")
    return payload["tasks"]


def _reset_task(env_base_url: str, task_id: str) -> Observation:
    payload = requests.post(f"{env_base_url}/reset", json={"task_id": task_id}, timeout=30).json()
    return Observation.model_validate(payload)


def _step_task(env_base_url: str, action: Dict) -> Dict:
    return requests.post(f"{env_base_url}/step", json=action, timeout=30).json()


def _grade_task(env_base_url: str, task_id: str) -> Dict:
    return requests.post(f"{env_base_url}/grader", json={"task_id": task_id}, timeout=30).json()


def _choose_action(client: Any, model_name: str, task: Dict, observation: Observation, step_count: int) -> Dict:
    fallback_action = choose_action(task, observation, step_count)
    prompt = (
        "You are an API debugging and repair agent in an OpenEnv benchmark.\n"
        "Return only a JSON object with keys: action_type, content, and optionally predicted_diagnosis "
        "(required when action_type is analyze).\n"
        "action_type must be one of: analyze, ask, propose_fix, apply_fix, confirm_done.\n"
        "predicted_diagnosis must be one of: missing_required_field, wrong_request_line, upstream_or_ambiguous.\n"
        "Rules: If the task requires clarification, use ask first with a concrete question. "
        "Use analyze with the correct predicted_diagnosis once you have enough context. "
        "propose_fix must include all remediation details needed for the scenario. "
        "apply_fix only after a complete proposal. Use confirm_done only when the task requires it (hard task).\n"
        f"Task JSON: {json.dumps(task, ensure_ascii=True)}\n"
        f"Observation JSON: {observation.model_dump_json()}\n"
        f"Step index: {step_count}\n"
    )

    try:
        response = client.responses.create(model=model_name, input=prompt)
        action = _extract_json_object(response.output_text)
        return Action.model_validate(action).model_dump(exclude_none=True)
    except Exception:
        return fallback_action


def main() -> None:
    warnings.filterwarnings("ignore")

    api_base_url = _get_env_with_default("API_BASE_URL", "https://api.openai.com/v1")
    model_name = _get_env_with_default("MODEL_NAME", "gpt-4.1-mini")
    hf_token = _require_env("HF_TOKEN")
    local_image_name = os.getenv("LOCAL_IMAGE_NAME", "").strip()
    env_base_url = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860").strip()

    _ = local_image_name

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package must be installed to run inference.py") from exc

    client = OpenAI(api_key=hf_token, base_url=api_base_url)
    tasks = _fetch_tasks(env_base_url)
    run_results: list[Dict] = []

    for task in tasks:
        _log_start(task)
        observation = _reset_task(env_base_url, task["id"])
        rewards = []
        done = False
        step_count = 0

        while not done and step_count < task.get("max_steps", 8):
            action = _choose_action(client, model_name, task, observation, step_count)
            response = _step_task(env_base_url, action)
            observation = Observation.model_validate(response["observation"])
            reward = clamp_open_unit_interval(float(response["reward"]["score"]))
            rewards.append(reward)
            step_count += 1
            _log_step(task["id"], step_count, reward)
            done = response["done"]

        graded = _grade_task(env_base_url, task["id"])
        final_result = {
            "task_id": task["id"],
            "difficulty": task["difficulty"],
            "steps": step_count,
            "score": clamp_open_unit_interval(float(graded["score"])),
            "rewards": rewards,
        }
        _log_end(final_result)
        run_results.append(final_result)

    _log_results(run_results)


if __name__ == "__main__":
    main()
