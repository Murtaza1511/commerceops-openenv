---
title: Commerceops Openenv
emoji: 🏆
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: 'OpenEnv benchmark for commerce support, payment triage, and '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---
title: CommerceOps OpenEnv
short_description: OpenEnv benchmark for commerce support, payment triage, and fraud escalation
sdk: docker
python_version: "3.11"
app_port: 7860
suggested_hardware: cpu-basic
---

# CommerceOps OpenEnv

CommerceOps OpenEnv is a benchmark environment for evaluating AI agents on realistic commerce support workflows. Instead of testing generic conversation ability, it measures whether an agent can follow structured operational workflows such as account recovery, duplicate-charge triage, and fraud escalation.

The project is designed for the Scaler School of Technology x Meta PyTorch OpenEnv Hackathon Round 1 requirements: a real-world environment, typed models, multiple tasks, deterministic grading, shaped rewards, a root `inference.py`, Docker packaging, and deployability to Hugging Face Spaces.

## Why this project

Commerce support is a strong benchmark setting because it combines:

- customer communication
- issue classification
- workflow sequencing
- safety-sensitive decision making
- escalation judgment

This environment is intentionally not a toy chat app. The agent is expected to classify problems correctly, ask for clarification when needed, provide useful next steps, and escalate only when the workflow requires it.

## Benchmark tasks

The environment currently includes three tasks with increasing difficulty.

### 1. Merchant Account Recovery

- Difficulty: easy
- Goal: identify an account access issue caused by a 2FA device change

### 2. Duplicate Charge Triage

- Difficulty: medium
- Goal: classify a duplicate payment issue and provide safe next steps

### 3. Unauthorized Payment Escalation

- Difficulty: hard
- Goal: ask for clarification, classify fraud risk, provide immediate safety guidance, and escalate correctly

## How the environment works

This project follows a standard RL-style environment loop:

1. `reset()` starts a fresh episode for a selected task
2. the agent sends an `Action`
3. `step()` updates the internal `State`
4. the environment returns:
   - `Observation`
   - `Reward`
   - `done`
   - `info`
5. the episode continues until completion or max steps
6. `grader()` computes the final deterministic score

## Project structure

- [app/main.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/app/main.py): FastAPI app entrypoint
- [app/api/routes.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/app/api/routes.py): HTTP routes for the OpenEnv API
- [app/models/schemas.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/app/models/schemas.py): Pydantic models for `Action`, `Observation`, `State`, and `Reward`
- [app/env/tasks.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/app/env/tasks.py): benchmark task definitions
- [app/env/environment.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/app/env/environment.py): core environment logic and reward shaping
- [app/env/grader.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/app/env/grader.py): deterministic task scoring
- [app/baseline_runner.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/app/baseline_runner.py): baseline control loop for local and HTTP evaluation
- [baseline.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/baseline.py): local baseline runner
- [inference.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/inference.py): Round 1 submission runner
- [client.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/client.py): small HTTP client helper
- [openenv.yaml](/Users/murtazatinwala/Developer/code/python/customer-support-env/openenv.yaml): environment manifest
- [Dockerfile](/Users/murtazatinwala/Developer/code/python/customer-support-env/Dockerfile): Docker deployment config

## API endpoints

The environment exposes the following routes:

- `GET /`
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`

`/tasks` returns both the available tasks and the JSON schema for the `Action` model.

## Local setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Start the API locally:

```bash
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Quick checks:

```bash
curl http://127.0.0.1:7860/
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/tasks
```

## Running the baseline

Run the local baseline policy:

```bash
./venv/bin/python baseline.py
```

This runs the built-in policy across all benchmark tasks and prints rewards and final scores.

## Submission inference script

The root [inference.py](/Users/murtazatinwala/Developer/code/python/customer-support-env/inference.py) is the Round 1 submission entrypoint. It:

- reads `API_BASE_URL`
- reads `MODEL_NAME`
- reads `HF_TOKEN`
- fetches tasks from the environment
- uses the OpenAI client for action generation
- emits `[START]`, `[STEP]`, and `[END]` logs

Example local run:

```bash
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4.1-mini \
HF_TOKEN=your_api_key \
ENV_BASE_URL=http://127.0.0.1:7860 \
./venv/bin/python inference.py
```

## Testing

Run the unit tests:

```bash
./venv/bin/python -m unittest test_models.py test_env.py test_api.py
```

Before submission, also verify:

- the root route returns HTTP `200`
- `/health` returns success
- `/tasks` returns three tasks and the action schema
- `inference.py` completes end-to-end
- rewards and final scores stay in the `0.0` to `1.0` range

## Docker

Build the image:

```bash
docker build -t commerceops-openenv .
```

Run it:

```bash
docker run -p 7860:7860 commerceops-openenv
```

Then test:

```bash
curl http://127.0.0.1:7860/
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/tasks
```

## Hugging Face Spaces deployment

This repository is configured for a Docker Space.

Recommended Space settings:

- SDK: `Docker`
- Visibility: `Public`
- Hardware: `CPU Basic`
- Port: `7860`

After creating the Space, push this repository to:

```bash
https://huggingface.co/spaces/<your-username>/<your-space-name>
```

Add these runtime secrets:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Then verify the deployed Space URL:

```bash
curl https://<your-space>.hf.space/
curl https://<your-space>.hf.space/health
curl https://<your-space>.hf.space/tasks
```

## Final submission checklist

Before submitting the Space URL on the hackathon dashboard, confirm:

- the Space is public
- the Space finishes building successfully
- the root Space URL returns HTTP `200`
- `inference.py` works against the deployed Space
- the repository contains `openenv.yaml`, `Dockerfile`, and root `inference.py`
- the README clearly explains the project and how to run it

## Notes

- If you use an OpenAI API key, store it as a Hugging Face Space secret
- Do not expose your API key in terminal screenshots, commits, or logs
- If an LLM response is malformed, the submission runner falls back to a benchmark-safe policy so the run remains stable

## References

- [Scaler OpenEnv Hackathon Dashboard](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard)
- [Meta PyTorch OpenEnv Hackathon Overview](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)
- [Hugging Face Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview)
- [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-docker)
