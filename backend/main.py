from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# IMPORT BOTH MODULES
from lost_found_ai.app import router as lost_found_router
from keyword_detection_backend.keyword_detection import router as keyword_router

# ✅ ADD THIS
from abusive_detection.app import router as abusive_router

app = FastAPI(title="Campus Guard Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    
)

# INCLUDE ALL ROUTERS
app.include_router(lost_found_router)
app.include_router(keyword_router)

# ✅ ADD THIS
app.include_router(abusive_router, prefix="/abuse", tags=["Abuse Detection"])

# ROOT
@app.get("/")
def home():
    return {"status": "Backend running"}