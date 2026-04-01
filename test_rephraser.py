from backend.services.rephraser_service import rephraser
import os

# Ensure we're in the right directory
os.chdir(r"c:\Users\Amodh\OneDrive\Desktop\projects\NLP_project")

def test_rephraser():
    examples = [
        "I hate you, you idiot",
        "Go away, nobody likes you",
        "You are trash",
        "Shut up and leave me alone"
    ]
    
    print("Testing Rephraser Service:")
    for text in examples:
        try:
            rephrased = rephraser.rephrase(text)
            print(f"Original: '{text}'")
            print(f"Rephrased: '{rephrased}'")
            print("-" * 20)
        except Exception as e:
            print(f"Error for '{text}': {e}")

if __name__ == "__main__":
    test_rephraser()
