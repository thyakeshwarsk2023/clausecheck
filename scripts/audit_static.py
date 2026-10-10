import json, os, sys

print('=== AUDIT 1A: DIRECTORY STRUCTURE INSPECTION ===')
required = [
    'data/intents.json',
    'models/intent_model.pth',
    'models/metadata.json',
    'reports/training_curves.png',
    'reports/confusion_matrix.png',
    'scripts/step1_build_dataset.py',
    'scripts/step2_train.py',
    'scripts/audit_test_suite.py',
    'app.py',
    'requirements.txt'
]
for f in required:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f'  [OK] {f:<45} ({size:,} bytes)')
    else:
        print(f'  [MISSING] {f}')

print()
print('=== AUDIT 1B: DATASET INTEGRITY ===')
with open('data/intents.json', 'r', encoding='utf-8') as fh:
    ds = json.load(fh)
intents = ds['intents']
print(f'  Total intent classes: {len(intents)}')
total = 0
required_keys = ['tag','regulatory_anchor','audit_title','audit_verdict','explanation','statutory_remedy','patterns']
for intent in intents:
    n = len(intent['patterns'])
    total += n
    missing_keys = [k for k in required_keys if k not in intent]
    schema_ok = 'OK' if not missing_keys else f'MISSING KEYS: {missing_keys}'
    tag = intent['tag']
    print(f'  [{schema_ok}] {tag:<34}: {n} patterns')
print(f'  Total patterns: {total}')
print(f'  Balance note: original spec was 50/class (300). Expanded to ~62/class (370) intentionally for ASR resilience.')

print()
print('=== AUDIT 1C: METADATA.JSON SCHEMA & TENSOR ALIGNMENT ===')
with open('models/metadata.json', 'r', encoding='utf-8') as fh:
    meta = json.load(fh)
for k, v in meta.items():
    if k not in ['audit_cards']:
        print(f'  {k}: {v}')
dim_ok = (
    meta.get('input_dim') == 384
    and meta.get('hidden_dim1') == 64
    and meta.get('hidden_dim2') == 32
    and meta.get('num_classes') == 6
)
audit_cards_count = len(meta.get('audit_cards', {}))
label_count = len(meta.get('label_names', []))
print(f'  Tensor arch (384->64->32->6): {"ALIGNED" if dim_ok else "MISMATCH"}')
print(f'  label_names count: {label_count} | audit_cards count: {audit_cards_count}')
print(f'  Labels match cards: {"YES" if label_count == audit_cards_count else "MISMATCH"}')

print()
print('=== AUDIT 1D: REQUIREMENTS.TXT CONTENTS ===')
with open('requirements.txt', 'r') as fh:
    for line in fh:
        print(f'  {line.rstrip()}')

print()
print('=== AUDIT 2A: MODEL CHECKPOINT LOAD TEST ===')
import torch
import torch.nn as nn

class ClauseCheckClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim1=64, hidden_dim2=32, num_classes=6, dropout_rate=0.3):
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

model = ClauseCheckClassifier(384, 64, 32, 6)
state = torch.load('models/intent_model.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()
num_params = sum(p.numel() for p in model.parameters())
model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f'  Load status: SUCCESS (no weight mismatch)')
print(f'  Total parameters: {num_params:,}')
print(f'  Model weight size: {model_bytes / 1024:.1f} KB  ({model_bytes / 1024 / 1024:.2f} MB)')
print(f'  All layers frozen for inference: {not any(p.requires_grad for p in model.parameters())}')

print()
print('=== AUDIT 2B: DUMMY INFERENCE SHAPE TEST ===')
import torch.nn.functional as F
test_input = torch.randn(1, 384)
with torch.no_grad():
    out = model(test_input)
    probs = F.softmax(out, dim=1)
    conf, idx = torch.max(probs, dim=1)
print(f'  Input shape:  {list(test_input.shape)}')
print(f'  Output shape: {list(out.shape)}')
print(f'  Softmax probabilities sum: {probs.sum().item():.6f}  (expected 1.0)')
print(f'  Dummy prediction index: {idx.item()} | confidence: {conf.item():.4f}')
print(f'  Shape alignment: {"PASS" if out.shape == (1, 6) else "FAIL"}')

print()
print('=== AUDIT 2C: FILE SIZE & MEMORY COMPLIANCE ===')
import os
pth_size_mb = os.path.getsize('models/intent_model.pth') / 1024 / 1024
print(f'  intent_model.pth on disk: {pth_size_mb:.2f} MB')
print(f'  120 MB compliance check: {"PASS" if pth_size_mb < 120 else "FAIL"}')
