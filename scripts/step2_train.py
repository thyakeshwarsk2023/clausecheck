import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure deterministic execution across runs
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DATA_FILE = "data/intents.json"
MODEL_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATASET & EXTRACT DENSE VECTORS
# ---------------------------------------------------------
print("[*] Ingesting ClauseCheck dataset from", DATA_FILE)
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)["intents"]

texts = []
labels = []
label_names = []
audit_cards = {}

for idx, intent in enumerate(raw_data):
    tag = intent["tag"]
    label_names.append(tag)
    audit_cards[tag] = {
        "regulatory_anchor": intent["regulatory_anchor"],
        "audit_title": intent["audit_title"],
        "audit_verdict": intent["audit_verdict"],
        "explanation": intent["explanation"],
        "statutory_remedy": intent["statutory_remedy"]
    }
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(idx)

labels = np.array(labels)
num_classes = len(label_names)

print(f"[*] Generating 384-dimensional embeddings via 'all-MiniLM-L6-v2' for {len(texts)} samples...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# ---------------------------------------------------------
# 2. STRATIFIED TRAIN / TEST SPLIT (80% Train, 20% Test)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
)
print(f"[+] Train Set: {len(X_train)} samples | Test Set: {len(X_test)} samples (Equally balanced across {num_classes} classes)")

class IntentDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_loader = DataLoader(IntentDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(IntentDataset(X_test, y_test), batch_size=16, shuffle=False)

# ---------------------------------------------------------
# 3. DEEP LEARNING CLASSIFIER ARCHITECTURE
# ---------------------------------------------------------
class ClauseCheckClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim1=64, hidden_dim2=32, num_classes=6, dropout_rate=0.3):
        super(ClauseCheckClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = ClauseCheckClassifier(input_dim=384, hidden_dim1=64, hidden_dim2=32, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)

# ---------------------------------------------------------
# 4. TRAINING & VALIDATION LOOP
# ---------------------------------------------------------
EPOCHS = 60
history = {"train_loss": [], "test_acc": []}

print("\n[*] Training Deep Learning Classifier...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total
    history["test_acc"].append(acc)

    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        print(f"    Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {avg_train_loss:.4f} | Validation Accuracy: {acc:.2%}")

# ---------------------------------------------------------
# 5. PERSIST CHECKPOINT & OPERATIONAL METADATA
# ---------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "intent_model.pth")
meta_path = os.path.join(MODEL_DIR, "metadata.json")

torch.save(model.state_dict(), model_path)

metadata = {
    "label_names": label_names,
    "audit_cards": audit_cards,
    "input_dim": 384,
    "hidden_dim1": 64,
    "hidden_dim2": 32,
    "num_classes": num_classes,
    "embedding_model": "all-MiniLM-L6-v2"
}

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print(f"\n[+] Trained Model Weights Exported : {model_path}")
print(f"[+] Metadata & XAI Schema Exported : {meta_path}")

# ---------------------------------------------------------
# 6. EVALUATION REPORTS & VISUALIZATIONS
# ---------------------------------------------------------
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(batch_y.cpu().numpy())

print("\n" + "=" * 65)
print(" CLAUSECHECK VALIDATION CLASSIFICATION REPORT (Held-Out 20%)")
print("=" * 65)
print(classification_report(all_trues, all_preds, target_names=label_names, digits=4, zero_division=0))
print("=" * 65)

# Plot 1: Training Dynamics
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history["train_loss"], color="crimson", lw=2)
plt.title("Cross-Entropy Loss Trajectory")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), history["test_acc"], color="teal", lw=2)
plt.title("Held-Out Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.0, 1.05])
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "training_curves.png"), dpi=300)
plt.close()

# Plot 2: Confusion Matrix
cm = confusion_matrix(all_trues, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.title("ClauseCheck Intent Classifier — Confusion Matrix")
plt.xlabel("Predicted Intent")
plt.ylabel("Ground Truth")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

print(f"[✓] Artifacts saved to '{REPORTS_DIR}/training_curves.png' and '{REPORTS_DIR}/confusion_matrix.png'")