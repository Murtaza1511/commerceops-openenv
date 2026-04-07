from typing import Any, Dict

import requests


class CommerceOpsClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/health", timeout=30).json()

    def tasks(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/tasks", timeout=30).json()

    def reset(self, task_id: str) -> Dict[str, Any]:
        return requests.post(f"{self.base_url}/reset", json=task_id, timeout=30).json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return requests.post(f"{self.base_url}/step", json=action, timeout=30).json()

    def state(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/state", timeout=30).json()

    def grader(self, task_id: str) -> Dict[str, Any]:
        return requests.post(f"{self.base_url}/grader", params={"task_id": task_id}, timeout=30).json()

    def baseline(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/baseline", timeout=30).json()
