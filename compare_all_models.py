"""
Comprehensive Model Comparison: LR, RF, LSTM, GNN, and GNN+LSTM Hybrid
Generates detailed classification reports and CSV outputs for all models
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from models import GNN_LSTM_Classifier
from utils import make_fully_connected_adj

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("Models: Logistic Regression, Random Forest, LSTM, GNN, GNN+LSTM Hybrid")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[1/7] Loading and preprocessing data...")
data = np.load('data/processed.npz')
X_all = data['X']  # Shape: (N, seq_len, features)
y_all = data['y']  # Shape: (N,)

print(f"Total sequences: {X_all.shape[0]}")
print(f"Sequence length: {X_all.shape[1]}")
print(f"Features per timestep: {X_all.shape[2]}")
print(f"Fraud rate: {y_all.sum()/len(y_all)*100:.2f}%")

# Split data
X_train_seq, X_test_seq, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Flatten sequences for traditional ML models (LR, RF)
X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

print(f"Train samples: {len(X_train_seq)} | Test samples: {len(X_test_seq)}")
print(f"Flattened feature dimension: {X_train_flat.shape[1]}")

# ============================================================================
# 2. LOGISTIC REGRESSION
# ============================================================================
print("\n[2/7] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_flat, y_train)

lr_pred = lr_model.predict(X_test_flat)
lr_proba = lr_model.predict_proba(X_test_flat)[:, 1]

print("Logistic Regression trained successfully!")

# ============================================================================
# 3. RANDOM FOREST
# ============================================================================
print("\n[3/7] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train_flat, y_train)

rf_pred = rf_model.predict(X_test_flat)
rf_proba = rf_model.predict_proba(X_test_flat)[:, 1]

print("Random Forest trained successfully!")

# ============================================================================
# 4. LSTM-ONLY MODEL
# ============================================================================
print("\n[4/7] Training LSTM-only model...")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last timestep
        return self.fc(last_hidden).squeeze(-1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

lstm_model = LSTMClassifier(input_size=X_train_seq.shape[2], hidden_size=64, num_layers=2).to(device)

# Prepare data loaders
train_dataset = TensorDataset(
    torch.from_numpy(X_train_seq).float(),
    torch.from_numpy(y_train).float()
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test_seq).float(),
    torch.from_numpy(y_test).float()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train LSTM
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Calculate class weights for balanced training
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
criterion_weighted = nn.BCELoss(weight=None)

lstm_model.train()
for epoch in range(5):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = lstm_model(xb)
        
        # Apply class weighting manually
        loss = criterion(pred, yb)
        weights = torch.where(yb == 1, pos_weight, 1.0)
        loss = (loss * weights).mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 2 == 0:
        print(f"  Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate LSTM
lstm_model.eval()
lstm_preds = []
lstm_probas = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        proba = lstm_model(xb).cpu().numpy()
        lstm_probas.extend(proba)
        lstm_preds.extend((proba > 0.5).astype(int))

lstm_pred = np.array(lstm_preds)
lstm_proba = np.array(lstm_probas)

print("LSTM model trained successfully!")

# ============================================================================
# 5. GNN-ONLY MODEL
# ============================================================================
print("\n[5/7] Training GNN-only model...")

class GNNClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        from models import SimpleGNNLayer
        self.layers = nn.ModuleList([
            SimpleGNNLayer(feat_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, adj):
        # x: (batch, seq_len, features)
        # adj: (batch, seq_len, seq_len)
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        # Global pooling: mean over all nodes/timesteps
        h_pooled = h.mean(dim=1)  # (batch, hidden_dim)
        return self.fc(h_pooled).squeeze(-1)

gnn_model = GNNClassifier(feat_dim=X_train_seq.shape[2], hidden_dim=64, num_layers=2).to(device)

# Train GNN
optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=0.001)

gnn_model.train()
for epoch in range(5):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        batch_size, seq_len = xb.shape[0], xb.shape[1]
        
        # Create adjacency matrix
        adj = make_fully_connected_adj(batch_size, seq_len, device=device)
        
        optimizer_gnn.zero_grad()
        pred = gnn_model(xb, adj)
        
        # Apply class weighting
        loss = criterion(pred, yb)
        weights = torch.where(yb == 1, pos_weight, 1.0)
        loss = (loss * weights).mean()
        
        loss.backward()
        optimizer_gnn.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 2 == 0:
        print(f"  Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate GNN
gnn_model.eval()
gnn_preds = []
gnn_probas = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        batch_size, seq_len = xb.shape[0], xb.shape[1]
        adj = make_fully_connected_adj(batch_size, seq_len, device=device)
        
        proba = gnn_model(xb, adj).cpu().numpy()
        gnn_probas.extend(proba)
        gnn_preds.extend((proba > 0.5).astype(int))

gnn_pred = np.array(gnn_preds)
gnn_proba = np.array(gnn_probas)

print("GNN model trained successfully!")

# ============================================================================
# 6. GNN+LSTM HYBRID MODEL (Load pretrained)
# ============================================================================
print("\n[6/7] Loading pretrained GNN+LSTM Hybrid model...")

hybrid_model = GNN_LSTM_Classifier(
    feat_dim=X_test_seq.shape[2],
    gnn_hidden=32,
    gnn_out=32,
    lstm_hidden=32,
    num_layers=1
).to(device)

# Try to load baseline model
try:
    hybrid_model.load_state_dict(torch.load('outputs/baseline_model.pt', map_location=device))
    print("Loaded baseline_model.pt")
except:
    print("Baseline model not found, training new hybrid model...")
    optimizer_hybrid = torch.optim.Adam(hybrid_model.parameters(), lr=0.001)
    
    hybrid_model.train()
    for epoch in range(5):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            batch_size, seq_len = xb.shape[0], xb.shape[1]
            adj = make_fully_connected_adj(batch_size, seq_len, device=device)
            
            optimizer_hybrid.zero_grad()
            pred = hybrid_model(xb, adj)
            
            loss = criterion(pred, yb)
            weights = torch.where(yb == 1, pos_weight, 1.0)
            loss = (loss * weights).mean()
            
            loss.backward()
            optimizer_hybrid.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate Hybrid
hybrid_model.eval()
hybrid_preds = []
hybrid_probas = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        batch_size, seq_len = xb.shape[0], xb.shape[1]
        adj = make_fully_connected_adj(batch_size, seq_len, device=device)
        
        proba = hybrid_model(xb, adj).cpu().numpy()
        hybrid_probas.extend(proba)
        hybrid_preds.extend((proba > 0.5).astype(int))

hybrid_pred = np.array(hybrid_preds)
hybrid_proba = np.array(hybrid_probas)

print("GNN+LSTM Hybrid model evaluated successfully!")

# ============================================================================
# 7. GENERATE CLASSIFICATION REPORTS AND ANALYSIS
# ============================================================================
print("\n[7/7] Generating comprehensive reports...")

def evaluate_model(name, y_true, y_pred, y_proba):
    """Generate comprehensive evaluation metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'True_Positives': int(tp),
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'predictions': y_pred,
        'probabilities': y_proba,
        'confusion_matrix': cm
    }

# Evaluate all models
models_results = [
    evaluate_model('Logistic Regression', y_test, lr_pred, lr_proba),
    evaluate_model('Random Forest', y_test, rf_pred, rf_proba),
    evaluate_model('LSTM-only', y_test, lstm_pred, lstm_proba),
    evaluate_model('GNN-only', y_test, gnn_pred, gnn_proba),
    evaluate_model('GNN+LSTM Hybrid', y_test, hybrid_pred, hybrid_proba)
]

# Print classification reports
print("\n" + "="*80)
print("CLASSIFICATION REPORTS")
print("="*80)

for result in models_results:
    name = result['Model']
    y_pred = result['predictions']
    
    print(f"\n{'='*80}")
    print(f"{name.upper()}")
    print(f"{'='*80}")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4, zero_division=0))
    
    print(f"\nDetailed Metrics:")
    print(f"  Accuracy:    {result['Accuracy']:.4f} ({result['Accuracy']*100:.2f}%)")
    print(f"  Precision:   {result['Precision']:.4f}")
    print(f"  Recall:      {result['Recall']:.4f}")
    print(f"  F1-Score:    {result['F1-Score']:.4f}")
    print(f"  Specificity: {result['Specificity']:.4f}")
    print(f"  AUC-ROC:     {result['AUC-ROC']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = result['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    print(f"                Predicted")
    print(f"              Normal  Fraud")
    print(f"Actual Normal  {tn:5d}  {fp:5d}")
    print(f"       Fraud   {fn:5d}  {tp:5d}")

# ============================================================================
# 8. SAVE RESULTS TO CSV
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS TO CSV")
print("="*80)

os.makedirs('outputs', exist_ok=True)

# Summary CSV
summary_df = pd.DataFrame([{
    'Model': r['Model'],
    'Accuracy': r['Accuracy'],
    'Precision': r['Precision'],
    'Recall': r['Recall'],
    'F1-Score': r['F1-Score'],
    'Specificity': r['Specificity'],
    'AUC-ROC': r['AUC-ROC'],
    'True_Positives': r['True_Positives'],
    'True_Negatives': r['True_Negatives'],
    'False_Positives': r['False_Positives'],
    'False_Negatives': r['False_Negatives']
} for r in models_results])

summary_df.to_csv('outputs/all_models_comparison.csv', index=False)
print("Saved: outputs/all_models_comparison.csv")

# Detailed predictions for each model
for result in models_results:
    name = result['Model'].lower().replace(' ', '_').replace('+', '_')
    pred_df = pd.DataFrame({
        'Index': range(len(result['predictions'])),
        'True_Label': y_test,
        'Predicted_Label': result['predictions'],
        'Predicted_Probability': result['probabilities'],
        'Correct': (result['predictions'] == y_test).astype(int)
    })
    pred_df.to_csv(f'outputs/{name}_predictions.csv', index=False)
    print(f"Saved: outputs/{name}_predictions.csv")

# ============================================================================
# 9. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Figure 1: Metrics Comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC-ROC']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    values = [r[metric] for r in models_results]
    model_names = [r['Model'] for r in models_results]
    
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/all_models_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/all_models_metrics_comparison.png")

# Figure 2: Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')

for idx, result in enumerate(models_results):
    ax = axes[idx // 3, idx % 3]
    cm = result['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'],
               cbar_kws={'label': 'Count'})
    ax.set_title(f"{result['Model']}\nAUC: {result['AUC-ROC']:.4f}", fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')

# Hide the last subplot (we have 5 models)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/all_models_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/all_models_confusion_matrices.png")

# Figure 3: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for result, color in zip(models_results, colors):
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    auc = result['AUC-ROC']
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{result['Model']} (AUC = {auc:.4f})", alpha=0.8)

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('outputs/all_models_roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/all_models_roc_curves.png")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print("\nMODEL RANKINGS BY AUC-ROC:")
ranked = sorted(models_results, key=lambda x: x['AUC-ROC'], reverse=True)
for i, result in enumerate(ranked, 1):
    print(f"  {i}. {result['Model']:<25} AUC: {result['AUC-ROC']:.4f}")

print("\nOUTPUT FILES GENERATED:")
print("  - all_models_comparison.csv")
print("  - logistic_regression_predictions.csv")
print("  - random_forest_predictions.csv")
print("  - lstm_only_predictions.csv")
print("  - gnn_only_predictions.csv")
print("  - gnn_lstm_hybrid_predictions.csv")
print("  - all_models_metrics_comparison.png")
print("  - all_models_confusion_matrices.png")
print("  - all_models_roc_curves.png")

print("\nAll outputs saved to: outputs/")
print("Ready for presentation!")
print("="*80)
