import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import random
import re
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ==========================
# 1. Fix random seeds for reproducibility
# ==========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # Set seed

# ==========================
# 2. Text preprocessing
# ==========================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Keep only letters, numbers and spaces
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# ==========================
# 3. Load dataset
# ==========================
def load_data():
    df = pd.read_csv("datasets/pytorch.csv", encoding="utf-8")
    df.dropna(inplace=True)

    df["cleaned_text"] = (df["Title"].astype(str) + " " + df["Body"].astype(str)).apply(preprocess_text)

    if "class" in df.columns:
        df["label"] = df["class"]
    elif "related" in df.columns:
        df["label"] = df["related"]
    else:
        raise ValueError("No label column found!")

    return df

# ==========================
# 4. Custom dataset
# ==========================
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================
# 5. Training function
# ==========================
def train_model(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))  # Make model focus more on class `1`
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.004)  # Weight Decay=0.004

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate validation metrics
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = (probs[:, 1] > 0.32).long()  # Prediction threshold 0.32
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Calculate six metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=1)
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f} ")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("-" * 50)

    return auc

# ==========================
# 6. BERT training
# ==========================
def train_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Fixed data split (random_state=42)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_dataset = TextClassificationDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = TextClassificationDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # batch_size=32
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    train_model(model, train_loader, val_loader, device, epochs=15)

if __name__ == "__main__":
    train_bert()