from backend.services.rephraser_service import rephraser

test_text = "you are bad"
result = rephraser.rephrase(test_text)
print(f"Input: {test_text}")
print(f"Result: {result}")
