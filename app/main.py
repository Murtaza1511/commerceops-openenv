from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="CommerceOps OpenEnv",
    description="AI environment for commerce support, payments, and trust-risk operations",
    version="1.0"
)

app.include_router(router)
