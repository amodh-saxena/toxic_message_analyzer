from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def test_prompts_base():
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    test_texts = ["you are bad", "you are stupid", "I hate your work"]
    
    # Tried these:
    # f"rephrase toxic to polite: {text}"
    
    for text in test_texts:
        print(f"\n--- Testing for: '{text}' ---")
        
        # 1. Few-shot (Usually best for T5)
        p1 = f"Transfer toxic to polite: 'you are ugly' -> 'your appearance is unique'; 'you are weak' -> 'you are gaining strength'; '{text}' -> "
        
        # 2. Simple but direct
        p2 = f"rewrite the following sentence to be extremely professional and polite: {text}"
        
        for p in [p1, p2]:
            inputs = tokenizer.encode(p, return_tensors="pt", max_length=128, truncation=True)
            outputs = model.generate(
                inputs, 
                max_length=50, 
                num_beams=4, 
                repetition_penalty=2.5, # Boost to prevent looping
                length_penalty=1.0, 
                early_stopping=True
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print(f"Prompt: {p[:60]}...")
            print(f"Result: {result}")

if __name__ == "__main__":
    test_prompts_base()
