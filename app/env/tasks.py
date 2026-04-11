TASKS = [
    {
        "id": "task_1",
        "name": "Missing JSON field",
        "description": "POST body omits a required field; fix the JSON payload.",
        "difficulty": "easy",
        "artifact": (
            "POST /v1/orders HTTP/1.1\n"
            "Host: api.example.com\n"
            "Content-Type: application/json\n\n"
            '{"sku":"WIDGET-A1"}'
        ),
        "expected_diagnosis": "missing_required_field",
        "valid_fix_markers": ['"qty"', '"sku"'],
        "requires_clarification": False,
        "requires_confirm": False,
        "max_steps": 5,
        "clarification_response": "",
    },
    {
        "id": "task_2",
        "name": "Wrong method and path",
        "description": "Client uses GET with query instead of POST to search with JSON body.",
        "difficulty": "medium",
        "artifact": (
            "GET /v1/orders?page=1 HTTP/1.1\n"
            "Host: api.example.com\n"
            "Accept: application/json"
        ),
        "expected_diagnosis": "wrong_request_line",
        "valid_fix_markers": ["post", "/v1/orders/search", "application/json"],
        "requires_clarification": False,
        "requires_confirm": False,
        "max_steps": 6,
        "clarification_response": "",
    },
    {
        "id": "task_3",
        "name": "Ambiguous upstream failure",
        "description": "502 with unclear blast radius; clarify environment, then stabilize client and escalate monitoring.",
        "difficulty": "hard",
        "artifact": (
            "POST /v1/checkout HTTP/1.1\n"
            "Host: api.example.com\n"
            "Content-Type: application/json\n\n"
            '{"cart_id":"c9"}'
        ),
        "expected_diagnosis": "upstream_or_ambiguous",
        "valid_fix_markers": ["timeout", "retry", "idempotency", "upstream"],
        "requires_clarification": True,
        "requires_confirm": True,
        "max_steps": 8,
        "clarification_response": (
            "Only staging is failing; production traffic is healthy. Error started after the load balancer config change."
        ),
    },
]
