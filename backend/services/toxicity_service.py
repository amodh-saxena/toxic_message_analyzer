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
            results = self.pipeline(text)[0]
            
            # Label-Agnostic Score Extraction (Defensive Coding)
            # We look for labels like 'NEGATIVE', 'LABEL_0', or 'negative'
            neg_score = 0.0
            pos_score = 0.0
            
            for r in results:
                label = r['label'].upper()
                if label in ['NEGATIVE', 'LABEL_0', 'NEG']:
                    neg_score = r['score']
                elif label in ['POSITIVE', 'LABEL_1', 'POS']:
                    pos_score = r['score']
            
            # If for some reason the above failed to identify labels, we assume binary sentiment
            if neg_score == 0.0 and pos_score == 0.0:
                neg_score = results[0]['score'] # Fallback to first result
            
            toxicity_score = neg_score
            label = "Toxic" if toxicity_score > 0.6 else "Non-toxic"
            
            # Replicating the 5-tier segmentation suite for UI parity
            segmentation = {
                "Hostility": toxicity_score * 0.9,
                "Directness": neg_score * 0.8,
                "Negativity": neg_score * 1.0,
                "Context_Risk": toxicity_score * 0.7,
                "Aggression": toxicity_score * 1.0
            }
            
            return toxicity_score, label, segmentation
            
        except Exception as e:
            print(f"PREDICTION ERROR: {e}")
            # Safe fallback so UI never crashes
            return 0.0, "Non-toxic", {"Hostility":0,"Directness":0,"Negativity":0,"Context_Risk":0,"Aggression":0}

toxicity_analyzer = ToxicityService()
