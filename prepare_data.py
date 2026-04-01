import pandas as pd
from datasets import load_dataset
import os

def prepare_data(filename="toxic_comments.csv", target_samples=3000):
    print(f"Fetching {target_samples} samples from the Jigsaw dataset...")
    
    # Let's try multiple common dataset paths to be robust
    datasets_to_try = [
        ("thesofakillers/jigsaw-toxic-comment-classification-challenge", "default"),
        ("jigsaw_toxicity_pred", "default"),
        ("google/jigsaw_toxicity_pred", "default")
    ]
    
    dataset = None
    for path, config in datasets_to_try:
        try:
            print(f"Trying {path}...")
            dataset = load_dataset(path, split=f"train[:{target_samples}]")
            if dataset:
                print(f"Successfully loaded {path}!")
                break
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
            
    if not dataset:
        raise RuntimeError("Could not load any version of the Jigsaw dataset from Hugging Face.")

    df = pd.DataFrame(dataset)
    
    # Jigsaw datasets usually have these labels: 
    # 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Select and keep all categories if they exist
    existing_cols = [c for c in toxicity_columns if c in df.columns]
    text_col = 'comment_text' if 'comment_text' in df.columns else 'text'
    
    if existing_cols:
        # Keep all labels + text
        df = df[[text_col] + existing_cols].rename(columns={text_col: 'comment_text'})
    else:
        # Fallback for binary label datasets
        label_col = 'label' if 'label' in df.columns else 'is_toxic'
        df = df[[text_col, label_col]].rename(columns={text_col: 'comment_text', label_col: 'toxic'})
        # Fill other columns with 0
        for col in ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
            df[col] = 0
            
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Generated precision-focused dataset with {len(df)} samples in {filename}")

if __name__ == "__main__":
    prepare_data()
