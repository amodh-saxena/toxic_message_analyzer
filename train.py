import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import re

# 1. Dataset Class
class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Use float for multi-label BCE loss
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# 2. Preprocessing
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def compute_metrics(pred):
    labels = pred.label_ids
    # Multi-label prediction: sigmoid + threshold (0.5)
    probs = 1 / (1 + np.exp(-pred.predictions))
    preds = (probs > 0.5).astype(int)
    
    # Simple binary indicator for 'any' toxicity
    any_toxic_label = (labels.sum(axis=1) > 0).astype(int)
    any_toxic_preds = (preds.sum(axis=1) > 0).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(any_toxic_label, any_toxic_preds, average='binary', zero_division=0)
    acc = accuracy_score(any_toxic_label, any_toxic_preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(csv_path="toxic_comments.csv", model_save_path="./model/"):
    print("Loading segmented data...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['comment_text'])
    df['comment_text'] = df['comment_text'].apply(clean_text)

    # 6 segments: toxic, severe_toxic, obscene, threat, insult, identity_hate
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    X = df['comment_text'].tolist()
    y = df[label_cols].values.tolist()

    print("Splitting data...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Tokenizing...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    train_dataset = ToxicDataset(train_encodings, train_labels)
    test_dataset = ToxicDataset(test_encodings, test_labels)

    print("Initializing multi-label model (6 categories)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # We use num_labels=6 for segmentation
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
    model.to(device)

    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training (Segmentation Phase)...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

    print(f"Saving multi-label model to {model_save_path}...")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    if not os.path.exists("./model/"):
        os.makedirs("./model/")
    train_model()
