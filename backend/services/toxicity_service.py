from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os

class ToxicityService:
    def __init__(self, model_path="./model/"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please run training first.")
        
        print(f"Loading toxicity model from {self.model_path}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        
        # Categories mapping based on train.py
        categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Use sigmoid for multi-label independent probabilities
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            segmentation = {cat: float(prob) for cat, prob in zip(categories, probs)}
            
            # Overall toxicity score is the max of any toxic category
            max_score = float(max(probs))
            
        label = "Toxic" if max_score > 0.5 else "Non-toxic"
        return max_score, label, segmentation

# Singleton instance
toxicity_analyzer = ToxicityService()
