---
title: ApiDebug OpenEnv
emoji: 🔧
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: OpenEnv API repair benchmark with deterministic grading
python_version: "3.11"
app_port: 7860
suggested_hardware: cpu-basic
---

# ApiDebug OpenEnv

This is our entry for the **Scaler × Meta PyTorch OpenEnv** track. We wanted something closer to what engineers actually do than a generic chatbot: you get a broken HTTP-style request (or logs), and the agent has to **diagnose**, sometimes **ask** for missing context, **propose** a concrete fix, **apply** it, and in the hardest scenario **confirm** the incident is closed. The grader is fully rule-based, so scores are reproducible and there’s no hand-wavy “vibes” scoring.

**Why this might matter to reviewers:** the action space is small and explicit (`analyze`, `ask`, `propose_fix`, `apply_fix`, `confirm_done`), observations carry the artifact plus simulator feedback, and final scores always land strictly inside `(0, 1)` as required.

## What’s inside

There are **three tasks** (easy → hard): missing JSON field, wrong method/path, and an ambiguous upstream case where you must clarify staging vs production before diagnosing. Details live in `app/env/tasks.py`. The environment implementation is `ApiRepairEnv` in `app/env/environment.py`; grading logic is in `app/env/grader.py`.

The **`/baseline`** route (and `baseline.py`) run a simple scripted policy so you can sanity-check the stack without calling an LLM. **`inference.py`** is what we submit for evaluation: it calls the model API you configure, and only falls back to the scripted policy if the model output isn’t valid JSON or doesn’t validate against our `Action` schema—so a bad network day doesn’t brick the whole run.

## Running it locally

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Sanity curls:

```bash
curl http://127.0.0.1:7860/
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/tasks
```

**Baseline** (needs the server above on 7860, or change `BASE_URL` in `baseline.py`):

```bash
./venv/bin/python baseline.py
```

**Tests** — we run the whole suite before every push:

```bash
./venv/bin/python -m unittest discover -s . -p 'test*.py'
```

**Docker**

```bash
docker build -t apidebug-openenv .
docker run -p 7860:7860 apidebug-openenv
```

We added a `.dockerignore` so Hugging Face builds don’t upload your whole `.venv` or `.git` history by accident.

## Inference (what the hackathon runs)

You need `HF_TOKEN` (we use it as the OpenAI-compatible API key), plus `API_BASE_URL` and `MODEL_NAME`. Point `ENV_BASE_URL` at wherever the env is listening (local `7860` or your Space URL).

```bash
export ENV_BASE_URL=http://127.0.0.1:7860
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export HF_TOKEN=your_key_here
./venv/bin/python inference.py
```

You should see lines starting with `[START]`, `[STEP]`, `[END]`, and a final `[RESULTS]` JSON blob. Don’t paste keys into issues or screenshots.

## Hugging Face Space (what we actually submitted)

We ship a **Docker** Space on port **7860**, public, with the same `Dockerfile` as here. In Space **Settings → Secrets**, we set `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` so optional in-Space runs can call the same inference script.

After each deploy, it’s worth hitting your Space root and `/health` in a browser or with `curl` to confirm the build went green. If the dashboard asks for a repo link, point them at this GitHub repo; if it asks for the Space URL, use the one Hugging Face shows on the Space page (hostname varies slightly by account).

## Renaming the repo or Space later

If you rename the GitHub repo or the HF Space slug, update your git remotes (`git remote set-url …`) and any bookmarks. Both platforms usually redirect old names for a while, but it’s cleaner to use the new URL everywhere.

## Files worth skimming

`app/main.py` boots FastAPI. `app/api/routes.py` wires `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline`. `openenv.yaml` is the manifest. Everything else is under `app/env/` and `app/models/`.

## Links

- [Hackathon dashboard](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard)
- [HF Docker Spaces docs](https://huggingface.co/docs/hub/en/spaces-sdks-docker)
- [OpenEnv on GitHub](https://github.com/meta-pytorch/OpenEnv)

Thanks for taking the time to review this.
