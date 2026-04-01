import json
import torch
from backend.services.toxicity_service import toxicity_analyzer
from backend.services.rephraser_service import rephraser
from backend.db import SessionLocal, init_db
from backend.models import Message

def debug():
    print("Initializing DB...")
    init_db()
    
    print("Loading model...")
    toxicity_analyzer.load_model()
    
    test_input = "you are a total idiot"
    print(f"Testing input: {test_input}")
    
    try:
        score, label, segmentation = toxicity_analyzer.predict(test_input)
        print(f"Prediction: {score}, {label}")
        print(f"Segmentation: {segmentation}")
        
        rephrased = rephraser.rephrase(test_input, score=score)
        print(f"Rephrased: {rephrased}")
        
        db = SessionLocal()
        new_msg = Message(
            user_input=test_input,
            context="",
            toxicity_score=score,
            prediction_label=label,
            rephrased_output=rephrased,
            segmentation_scores=json.dumps(segmentation)
        )
        db.add(new_msg)
        db.commit()
        print("Successfully committed to DB!")
        db.close()
    except Exception as e:
        print(f"ERROR DURING DEBUG: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
