from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from keyword_detection import router as keyword_router

app = FastAPI(title="Campus Guard Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(keyword_router)

@app.get("/")
def home():
    return {"status": "Backend running"}