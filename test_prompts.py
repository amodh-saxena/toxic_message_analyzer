from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def test_prompts():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    test_text = "you are bad"
    
    prompts = [
        # User's current prompt (which failed)
        f"You are an AI assistant that rewrites toxic messages into polite ones. Message: \"{test_text}\" Polite version:",
        # Shorter direct prompt
        f"rewrite politely: {test_text}",
        # Task-specific prompt
        f"rephrase toxic to polite: {test_text}",
        # Few-shot prompt
        f"toxic: you are stupid. polite: your idea could be improved. toxic: {test_text}. polite:"
    ]
    
    print(f"Testing prompts for: '{test_text}'")
    for p in prompts:
        inputs = tokenizer.encode(p, return_tensors="pt", max_length=128, truncation=True)
        outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {p[:50]}...")
        print(f"Result: {result}")
        print("-" * 20)

if __name__ == "__main__":
    test_prompts()
