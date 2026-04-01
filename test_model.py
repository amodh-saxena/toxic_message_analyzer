from backend.services.toxicity_service import toxicity_analyzer
import os

# Ensure we're in the right directory
os.chdir(r"c:\Users\Amodh\OneDrive\Desktop\projects\NLP_project")

def test_model():
    examples = [
        "I hate you, you idiot",
        "Hello, how are you today?",
        "Go away, nobody likes you",
        "Thank you for the help!",
        "You are trash",
        "Have a wonderful day"
    ]
    
    print("Testing Toxicity Model:")
    try:
        toxicity_analyzer.load_model()
        for text in examples:
            score, label = toxicity_analyzer.predict(text)
            print(f"Text: '{text}' | Score: {score:.4f} | Label: {label}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model()
