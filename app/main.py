from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="ApiDebug OpenEnv",
    description="HTTP/API debugging and repair benchmark with deterministic grading",
    version="1.0",
)

app.include_router(router)
