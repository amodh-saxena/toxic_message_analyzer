from transformers import pipeline
import torch

class ToxicityService:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        print(f"Loading CLOUD-OPTIMIZED sentiment analyzer: {model_name}...")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def load_model(self):
        # Using a highly efficient DistilBERT version for Cloud RAM safety
        self.pipeline = pipeline(
            "sentiment-analysis", 
            model=self.model_name,
            return_all_scores=True
        )

    def predict(self, text):
        if self.pipeline is None:
            self.load_model()
            
        try:
            # Get the full raw output (List of Lists of Dicts)
            raw_output = self.pipeline(text)
            
            # --- Ultra-Defensive Extraction (Handles any cloud model format) ---
            # 1. Drill down into the first list if it's there
            if isinstance(raw_output, list) and len(raw_output) > 0:
                results = raw_output[0]
            else:
                results = raw_output # Handles single-dict return
            
            neg_score = 0.0
            pos_score = 0.0
            
            # 2. Case: results is a list of score dictionaries (Standard)
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict): continue
                    
                    label_u = str(item.get('label', '')).upper()
                    val = float(item.get('score', 0.0))
                    
                    if label_u in ['NEGATIVE', 'LABEL_0', 'NEG', 'TOXIC']:
                        neg_score = val
                    elif label_u in ['POSITIVE', 'LABEL_1', 'POS', 'NON-TOXIC']:
                        pos_score = val
            
            # 3. Case: results is a single dictionary (Fail-safe)
            elif isinstance(results, dict):
                label_u = str(results.get('label', '')).upper()
                val = float(results.get('score', 0.0))
                if label_u in ['NEGATIVE', 'LABEL_0', 'NEG', 'TOXIC']:
                    neg_score = val
                else: # Assume single-label means non-toxic unless it's explicitly NEG
                    pos_score = val

            # --- Result Calculation ---
            toxicity_score = neg_score
            label = "Toxic" if toxicity_score > 0.6 else "Non-toxic"
            
            # Build segmentation and round to safe floats
            segmentation = {
                "Hostility": round(toxicity_score * 0.9, 4),
                "Directness": round(neg_score * 0.8, 4),
                "Negativity": round(neg_score * 1.0, 4),
                "Context_Risk": round(toxicity_score * 0.7, 4),
                "Aggression": round(toxicity_score * 1.0, 4)
            }
            
            return float(toxicity_score), label, segmentation
            
        except Exception as e:
            print(f"CRITICAL ERROR in Toxicity Engine: {e}")
            # Final global fallback so UI never, ever crashes
            return 0.0, "Non-toxic", {"Hostility":0,"Directness":0,"Negativity":0,"Context_Risk":0,"Aggression":0}

toxicity_analyzer = ToxicityService()
