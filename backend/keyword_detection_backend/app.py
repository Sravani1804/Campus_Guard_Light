from fastapi import FastAPI
from keyword_detection import router as keyword_router

app = FastAPI()

app.include_router(keyword_router)

@app.get("/")
def home():
    return {"status": "Backend is running"}