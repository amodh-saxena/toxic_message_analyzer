from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class RephraserService:
    def __init__(self, model_name="t5-base"):
        print(f"Loading professional rephraser model: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # EXPERT BENCHMARK EXAMPLES (User Provided)
        self.examples = {
            "tier1": {
                "input": "Thanks for your help!",
                "output": "Thank you for your support, I appreciate it."
            },
            "tier2": {
                "input": "This is not great",
                "output": "This could be improved further to meet expectations."
            },
            "tier3": {
                "input": "Wow nice job messing that up",
                "output": "I think there might have been an issue; let’s review it together."
            },
            "tier4": {
                "input": "You always do this wrong",
                "output": "There seems to be a recurring issue; let’s work on improving it."
            },
            "tier5": {
                "input": "This is terrible work",
                "output": "I believe we can improve this by refining a few aspects together."
            },
            "tier6": {
                "input": "You are completely useless",
                "output": "Let’s focus on improving collaboration and addressing the challenges constructively."
            },
            "tier7": {
                "input": "You are the worst person ever, I hate working with you",
                "output": "I sincerely apologize for any frustration caused and would like to work together more positively moving forward."
            }
        }

    def get_tier_key(self, score):
        if score <= 0.15: return "tier1"
        if score <= 0.30: return "tier2"
        if score <= 0.45: return "tier3"
        if score <= 0.60: return "tier4"
        if score <= 0.75: return "tier5"
        if score <= 0.90: return "tier6"
        return "tier7"

    def rephrase(self, text, context=None, score=0.0):
        tier_key = self.get_tier_key(score)
        benchmark = self.examples[tier_key]["output"]
        
        try:
            # Using the benchmark as a strong hint for the model
            instruction = f"Instruction: Rephrase the input using the following professional standard as a guide: '{benchmark}'"

            full_prompt = f"Task: {instruction} Input: {text}"
            if context:
                full_prompt = f"Context: {context} | {full_prompt}"
                
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            outputs = self.model.generate(
                inputs, 
                max_length=200, 
                num_beams=8, 
                repetition_penalty=1.5, 
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Post-cleanup
            for prefix in ["Positive Version:", "Polite Version:", "Result:", "Instruction:", "Supportive Version:"]:
                if prefix in decoded:
                    decoded = decoded.split(prefix)[-1].strip()
            
            # If AI fails to generate something unique or clear, we use the expert benchmark directly
            if not decoded or decoded.lower() == "false" or decoded.lower() == text.lower() or len(decoded) < 5:
                return benchmark
            
            return str(decoded)
            
        except Exception as e:
            print(f"REPHRASING ERROR: {e}")
            return benchmark

# Singleton instance
rephraser = RephraserService()
