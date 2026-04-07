from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ACTION MODEL

class Action(BaseModel):
    action_type: Literal[
        "classify",
        "ask",
        "respond",
        "resolve",
        "escalate",
    ] = Field(..., description="Type of action taken by the agent.")

    content: str = Field(..., description="Text content of the action")

    predicted_issue: Optional[Literal[
        "payment_issue",
        "account_access",
        "fraud_risk",
        "product_bug"
    ]] = Field(
        None,
        description="Predicted issue when classifying"
    )

    # Observation Model

class Observation(BaseModel):
    latest_customer_message: str
    conversation_history: List[str]
    steps_remaining: int
    sentiment: float
    task_id: Optional[str] = None
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "classify",
            "ask",
            "respond",
            "resolve",
            "escalate",
        ]
    )

# STATE MODEL

class State(BaseModel):
    ticket_id: str
    customer_query: str
    task_id: Optional[str] = None
    issue_type: Optional[str] = None
    conversation_history: List[str] = Field(default_factory=list)
    sentiment: float = 0.0
    resolved: bool = False
    steps_taken: int = 0
    max_steps: int = 5
    knowledge_used: List[str] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    matched_solution_keywords: List[str] = Field(default_factory=list)
    classification_correct: bool = False
    classification_attempts: int = 0
    asked_clarification: bool = False
    responded_helpfully: bool = False
    helpful_response_score: float = 0.0
    premature_resolution_attempts: int = 0


# REWARD MODEL

class Reward(BaseModel):
    score: float
