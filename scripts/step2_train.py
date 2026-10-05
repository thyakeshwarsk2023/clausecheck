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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DATA_FILE  = "data/intents.json"
MODEL_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)["intents"]

texts, labels, label_names, audit_cards = [], [], [], {}

for idx, intent in enumerate(raw_data):
    tag = intent["tag"]
    label_names.append(tag)
    audit_cards[tag] = {
        "regulatory_anchor": intent["regulatory_anchor"],
        "audit_title":       intent["audit_title"],
        "audit_verdict":     intent["audit_verdict"],
        "explanation":       intent["explanation"],
        "statutory_remedy":  intent["statutory_remedy"],
    }
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(idx)

labels     = np.array(labels)
num_classes = len(label_names)

print(f"Embedding {len(texts)} samples via all-MiniLM-L6-v2...")
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
)
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Classes: {num_classes}")


class IntentDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data,  dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ClauseCheckClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim1=64, hidden_dim2=32,
                 num_classes=6, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


train_loader = DataLoader(IntentDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader  = DataLoader(IntentDataset(X_test,  y_test),  batch_size=16, shuffle=False)

model     = ClauseCheckClassifier(input_dim=384, hidden_dim1=64, hidden_dim2=32, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)

EPOCHS    = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

history         = {"train_loss": [], "test_acc": [], "macro_f1": []}
best_f1         = 0.0
best_epoch      = 0
best_state_dict = None

model_path = os.path.join(MODEL_DIR, "intent_model.pth")
meta_path  = os.path.join(MODEL_DIR, "metadata.json")

print("Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    avg_loss = running_loss / len(train_loader)
    history["train_loss"].append(avg_loss)

    model.eval()
    correct, total = 0, 0
    preds_ep, trues_ep = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            out   = model(batch_x)
            preds = torch.argmax(out, dim=1)
            correct      += (preds == batch_y).sum().item()
            total        += batch_y.size(0)
            preds_ep.extend(preds.cpu().numpy())
            trues_ep.extend(batch_y.cpu().numpy())

    acc      = correct / total
    macro_f1 = f1_score(trues_ep, preds_ep, average="macro", zero_division=0)
    history["test_acc"].append(acc)
    history["macro_f1"].append(macro_f1)

    if macro_f1 > best_f1:
        best_f1         = macro_f1
        best_epoch      = epoch + 1
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch [{epoch+1:03d}/{EPOCHS}] loss={avg_loss:.4f} acc={acc:.2%} macro_f1={macro_f1:.4f}")

print(f"\nBest checkpoint: epoch {best_epoch}, macro_f1={best_f1:.4f}")
torch.save(best_state_dict, model_path)

metadata = {
    "label_names":    label_names,
    "audit_cards":    audit_cards,
    "input_dim":      384,
    "hidden_dim1":    64,
    "hidden_dim2":    32,
    "num_classes":    num_classes,
    "embedding_model": "all-MiniLM-L6-v2",
    "best_epoch":     best_epoch,
    "best_macro_f1":  round(best_f1, 6),
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print(f"Model  -> {model_path}")
print(f"Meta   -> {meta_path}")

model.load_state_dict(best_state_dict)
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        preds = torch.argmax(model(batch_x), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(batch_y.cpu().numpy())

print("\n" + "=" * 65)
print(" CLAUSECHECK VALIDATION CLASSIFICATION REPORT (Held-Out 20%)")
print("=" * 65)
print(classification_report(all_trues, all_preds, target_names=label_names, digits=4, zero_division=0))
print("=" * 65)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history["train_loss"], color="crimson", lw=2)
plt.axvline(best_epoch, color="gold", linestyle="--", lw=1.5, label=f"best epoch {best_epoch}")
plt.title("Cross-Entropy Loss Trajectory")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), history["test_acc"],  color="teal",   lw=2, label="Accuracy")
plt.plot(range(1, EPOCHS + 1), history["macro_f1"],  color="purple", lw=2, label="Macro F1")
plt.axvline(best_epoch, color="gold", linestyle="--", lw=1.5, label=f"best epoch {best_epoch}")
plt.title("Held-Out Validation Metrics")
plt.xlabel("Epoch")
plt.ylim([0.0, 1.05])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "training_curves.png"), dpi=300)
plt.close()

cm = confusion_matrix(all_trues, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("ClauseCheck Intent Classifier — Confusion Matrix")
plt.xlabel("Predicted Intent")
plt.ylabel("Ground Truth")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

print(f"Artifacts -> {REPORTS_DIR}/training_curves.png, confusion_matrix.png")