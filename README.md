---
title: ApiDebug OpenEnv
emoji: 🔧
colorFrom: blue
colorTo: slate
sdk: docker
pinned: false
license: mit
short_description: OpenEnv benchmark for HTTP/API debugging and deterministic repair grading
python_version: "3.11"
app_port: 7860
suggested_hardware: cpu-basic
---

# ApiDebug OpenEnv

ApiDebug OpenEnv evaluates AI agents on **structured API debugging and repair**: reading HTTP-shaped artifacts, diagnosing root causes, proposing fixes with required details, applying patches, and confirming closure when policy demands it. Scoring is **deterministic** (diagnosis match, clarification when required, fix marker coverage, clean resolution).

Built for the Scaler × Meta PyTorch OpenEnv Hackathon: typed Pydantic models, **3+** tasks (easy → hard), shaped step rewards, `/grader`, root `inference.py`, Docker, and Hugging Face Spaces.

## Evaluation methodology

- **`/baseline`** and `baseline.py` run a **scripted policy** (`choose_action` in [`app/baseline_runner.py`](app/baseline_runner.py)) for regression and sanity checks.
- **`inference.py`** is the **submission agent**: it calls the configured chat model and only **falls back** to the scripted policy if the model returns invalid JSON or fails validation—so runs stay stable without oracle-style comparison to the baseline on every step.

## Benchmark tasks

1. **Missing JSON field** (easy): invalid body; diagnose `missing_required_field`, propose a body containing required markers, `apply_fix`.
2. **Wrong request line** (medium): wrong method/path; diagnose `wrong_request_line`, propose POST + search path + JSON content type markers, `apply_fix`.
3. **Ambiguous upstream** (hard): 502 with unclear blast radius; **ask** for environment scope first, diagnose `upstream_or_ambiguous`, propose remediation markers (timeout, retry, idempotency, upstream), `apply_fix`, then **`confirm_done`**.

## Environment loop

1. `POST /reset` starts an episode for a task id.
2. `POST /step` accepts an `Action` (`analyze`, `ask`, `propose_fix`, `apply_fix`, `confirm_done`).
3. Response includes `observation`, `reward`, `done`, `info`.
4. `POST /grader` returns the final deterministic score in `(0, 1)`.

## Project structure

- [`app/main.py`](app/main.py): FastAPI entrypoint
- [`app/api/routes.py`](app/api/routes.py): OpenEnv HTTP API
- [`app/models/schemas.py`](app/models/schemas.py): `Action`, `Observation`, `State`, `Reward`
- [`app/env/tasks.py`](app/env/tasks.py): task definitions
- [`app/env/environment.py`](app/env/environment.py): `ApiRepairEnv`
- [`app/env/grader.py`](app/env/grader.py): deterministic grading
- [`app/baseline_runner.py`](app/baseline_runner.py): baseline loop
- [`baseline.py`](baseline.py): CLI baseline against local HTTP server
- [`inference.py`](inference.py): submission runner
- [`client.py`](client.py): minimal HTTP helper
- [`openenv.yaml`](openenv.yaml): manifest
- [`Dockerfile`](Dockerfile): container image

## API endpoints

- `GET /`, `GET /health`
- `POST /reset`, `POST /step`, `GET /state`
- `GET /tasks`, `POST /grader`, `GET /grader`, `GET /baseline`

## Local setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 7860
```

```bash
curl http://127.0.0.1:7860/
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/tasks
```

## Baseline

```bash
./venv/bin/python baseline.py
```

(Requires the API running on port 7860 unless you change `BASE_URL` in `baseline.py`.)

## Submission inference

```bash
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4.1-mini \
HF_TOKEN=your_api_key \
ENV_BASE_URL=http://127.0.0.1:7860 \
./venv/bin/python inference.py
```

## Testing

```bash
./venv/bin/python -m unittest test_models.py test_env.py test_api.py test_inference.py
```

## Docker

```bash
docker build -t apidebug-openenv .
docker run -p 7860:7860 apidebug-openenv
```

The [`.dockerignore`](.dockerignore) file excludes virtualenvs, `.git`, and caches so builds stay smaller and faster.

---

## Step-by-step: Hugging Face Space and resubmit

Do these in order on your machine and in the browser.

### Step 1 — Confirm the project works locally

```bash
cd /path/to/customer-support-env
./venv/bin/python -m unittest test_models.py test_env.py test_api.py test_inference.py
docker build -t apidebug-openenv .
```

Optional smoke test:

```bash
docker run --rm -d -p 7860:7860 --name apidebug-smoke apidebug-openenv
curl -s http://127.0.0.1:7860/health
docker stop apidebug-smoke
```

### Step 2 — Push this repository to GitHub (if it is not already)

- Create a repo or use an existing one.
- Push your branch (e.g. `main`) so Hugging Face can build from it, **or** use `git push` directly to Hugging Face (see Step 3).

### Step 3 — Create or open a Hugging Face Docker Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) → **Create new Space**.
2. Choose **Docker** as the SDK, set **port** to **7860**, visibility **Public** (required for many hackathon checks).
3. **Connect** the Space to your GitHub repository *or* add the Space as a remote and push:

   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push hf main
   ```

4. Wait until the Space **build** finishes (Build logs should show success).

### Step 4 — Add secrets (never commit keys)

In the Space: **Settings → Secrets and variables → Secrets**, add:

| Name | Example value |
|------|----------------|
| `HF_TOKEN` | Your LLM API key (used by `inference.py` as `api_key`) |
| `API_BASE_URL` | `https://api.openai.com/v1` (or your provider’s base URL) |
| `MODEL_NAME` | `gpt-4.1-mini` (or the model id your provider expects) |

`ENV_BASE_URL` is only needed when you run `inference.py` **pointing at** the Space from your laptop; the Space itself already serves the app on port 7860.

### Step 5 — Verify the deployed Space

Replace `<user>` and `<space>` with yours:

```bash
curl -sS "https://<user>-<space>.hf.space/" | head -c 300
curl -sS "https://<user>-<space>.hf.space/health"
curl -sS "https://<user>-<space>.hf.space/tasks" | head -c 500
```

Some Spaces use `hf.space` hostnames; use the **exact** URL shown on your Space page. You want HTTP **200** and JSON mentioning `apidebug-openenv`.

### Step 6 — Run `inference.py` against the live Space (from your laptop)

```bash
export ENV_BASE_URL="https://<your-space-url>"   # no trailing slash, or try as shown on HF
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export HF_TOKEN="sk-..."   # or paste from your secret store

./venv/bin/python inference.py
```

You should see `[START]`, `[STEP]`, `[END]`, and `[RESULTS]` with scores strictly between 0 and 1.

### Step 7 — Resubmit on the hackathon dashboard

1. Open the [Scaler hackathon dashboard](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard).
2. Submit or update your entry with the **public Space URL** and any **repository link** the form asks for.
3. Submit **before** the organizer’s deadline.

### Step 8 — If something fails

- **Build fails:** read the Space build log; fix `Dockerfile` / `requirements.txt` locally, push again.
- **`inference.py` errors:** confirm secrets exist, `ENV_BASE_URL` matches the live Space, and the model API is reachable from your network.
- **Discord / troubleshooting:** use the links in the organizer email.

## Submission checklist

- Public Space builds; `/` returns HTTP 200.
- Repo includes `openenv.yaml`, `Dockerfile`, root `inference.py`.
- Do not commit API keys; use Space secrets only.

## Renaming the GitHub repository and Hugging Face Space

Do this **after** a successful push, when you want URLs to match the ApiDebug project (example new name: `apidebug-openenv`).

### GitHub

1. Open the repo on GitHub → **Settings** → **General** → **Repository name**.
2. Rename (e.g. `commerceops-openenv` → `apidebug-openenv`) and confirm.
3. On your computer, point `github` at the new URL:

   ```bash
   git remote set-url github https://github.com/Murtaza1511/apidebug-openenv.git
   git fetch github
   ```

   Replace `Murtaza1511` / `apidebug-openenv` if your username or chosen name differs.

### Hugging Face Space

1. Open the Space → **Settings** (gear) → change the **Space name** (e.g. `apidebug-openenv`). Save.
2. Your Space URL becomes `https://huggingface.co/spaces/<user>/<new-space-name>`.
3. Update the `origin` remote:

   ```bash
   git remote set-url origin https://huggingface.co/spaces/Murtaza786/apidebug-openenv
   git fetch origin
   ```

   Replace `Murtaza786` and the space name with yours.

4. Update any **hackathon dashboard** links, **README** bookmarks, and **`ENV_BASE_URL`** if you hardcoded the old Space URL.

GitHub and Hugging Face keep **redirects** from old names for a while, but you should use the new URLs everywhere going forward.

## References

- [Scaler OpenEnv Hackathon Dashboard](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard)
- [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-docker)
- [Meta PyTorch OpenEnv](https://github.com/meta-pytorch/OpenEnv)
