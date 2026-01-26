from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lost_found_ai.app import router as lost_found_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lost_found_router)
