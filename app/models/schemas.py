from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    action_type: Literal[
        "analyze",
        "ask",
        "propose_fix",
        "apply_fix",
        "confirm_done",
    ] = Field(..., description="Type of action in the API repair episode.")

    content: str = Field(..., description="Text: diagnosis notes, question, proposed fix, or confirmation.")

    predicted_diagnosis: Optional[
        Literal[
            "missing_required_field",
            "wrong_request_line",
            "upstream_or_ambiguous",
        ]
    ] = Field(None, description="Root-cause label when action_type is analyze.")


class Observation(BaseModel):
    artifact: str = Field(..., description="Broken HTTP request or client snippet under repair.")
    latest_feedback: str = Field(..., description="Latest simulator or server feedback line.")
    conversation_history: List[str]
    steps_remaining: int
    sentiment: float = 0.0
    task_id: Optional[str] = None
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "analyze",
            "ask",
            "propose_fix",
            "apply_fix",
            "confirm_done",
        ]
    )


class State(BaseModel):
    request_id: str
    task_id: Optional[str] = None
    artifact: str = ""
    conversation_history: List[str] = Field(default_factory=list)
    sentiment: float = 0.0
    diagnosis_correct: bool = False
    diagnosis_attempts: int = 0
    asked_clarification: bool = False
    matched_fix_markers: List[str] = Field(default_factory=list)
    fix_proposed: bool = False
    fix_applied: bool = False
    resolved: bool = False
    steps_taken: int = 0
    max_steps: int = 8
    action_history: List[str] = Field(default_factory=list)
    premature_confirm_attempts: int = 0


class Reward(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0)
