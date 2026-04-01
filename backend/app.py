from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import sys
import json

from .db import SessionLocal, init_db
from .models import Message
from .services.toxicity_service import toxicity_analyzer
from .services.rephraser_service import rephraser

app = FastAPI(title="Toxic Message Analyzer")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class AnalysisRequest(BaseModel):
    user_input: str

class AnalysisResponse(BaseModel):
    id: int
    user_input: str
    context: str
    toxicity_score: float
    prediction_label: str
    rephrased_output: Optional[str] # REVERTED TO STRING
    segmentation_scores: Optional[dict]
    timestamp: datetime

@app.on_event("startup")
def startup_event():
    print("--- Backend Starting Up ---")
    init_db()
    try:
        toxicity_analyzer.load_model()
        print("--- Toxicity Model Loaded Successfully ---")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load toxicity model: {e}")

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_message(request: AnalysisRequest, db: Session = Depends(get_db)):
    # 1. Toxicity Prediction with Segmentation
    score, label, segmentation = toxicity_analyzer.predict(request.user_input)
    
    # 2. Context-Aware Tiered Rephrasing (7-Tier Logic)
    history = db.query(Message).order_by(Message.timestamp.desc()).limit(3).all()
    context = " | ".join([h.user_input for h in history]) if history else ""
    
    # Returns a SINGLE rephrased string based on 7-tier logic
    rephrased_str = rephraser.rephrase(request.user_input, context=context, score=score)
    
    # 3. Persist to DB
    new_msg = Message(
        user_input=request.user_input,
        context=context,
        toxicity_score=score,
        prediction_label=label,
        rephrased_output=rephrased_str,
        segmentation_scores=json.dumps(segmentation)
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    
    return {
        "id": new_msg.id,
        "user_input": new_msg.user_input,
        "context": new_msg.context,
        "toxicity_score": new_msg.toxicity_score,
        "prediction_label": new_msg.prediction_label,
        "rephrased_output": rephrased_str,
        "segmentation_scores": segmentation,
        "timestamp": new_msg.timestamp
    }

@app.get("/history", response_model=List[AnalysisResponse])
def get_history(db: Session = Depends(get_db)):
    messages = db.query(Message).order_by(Message.timestamp.desc()).all()
    
    response = []
    for m in messages:
        try:
            seg = json.loads(m.segmentation_scores) if m.segmentation_scores else None
        except:
            seg = None
            
        response.append({
            "id": m.id,
            "user_input": m.user_input,
            "context": m.context,
            "toxicity_score": m.toxicity_score,
            "prediction_label": m.prediction_label,
            "rephrased_output": m.rephrased_output,
            "segmentation_scores": seg,
            "timestamp": m.timestamp
        })
        
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
