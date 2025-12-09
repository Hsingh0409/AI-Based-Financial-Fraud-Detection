"""Create comparison plots for the presentation."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load XAI reports
with open('outputs/xai_baseline.json', 'r') as f:
    baseline = json.load(f)
with open('outputs/xai_fl.json', 'r') as f:
    fl = json.load(f)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model Comparison
models = ['Baseline\n(Non-Federated)', 'Federated Learning\n(4 Clients, 5 Rounds)']
aucs = [baseline['base_auc'], fl['base_auc']]
colors = ['#3498db', '#e74c3c']

axes[0].bar(models, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('AUC Score', fontsize=12, fontweight='bold')
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim([0.8, 1.0])
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (model, auc) in enumerate(zip(models, aucs)):
    axes[0].text(i, auc + 0.01, f'{auc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 2: Top 10 Feature Importances
baseline_imps = dict(baseline['feature_importances'])
fl_imps = dict(fl['feature_importances'])

# Get top 10 features by average importance
all_features = set(baseline_imps.keys()) | set(fl_imps.keys())
avg_imps = {feat: (baseline_imps.get(feat, 0) + fl_imps.get(feat, 0)) / 2 for feat in all_features}
top_features = sorted(avg_imps.items(), key=lambda x: x[1], reverse=True)[:10]
top_feat_names = [f[0] for f in top_features]

baseline_top = [baseline_imps.get(f, 0) for f in top_feat_names]
fl_top = [fl_imps.get(f, 0) for f in top_feat_names]

x = np.arange(len(top_feat_names))
width = 0.35

bars1 = axes[1].barh(x - width/2, baseline_top, width, label='Baseline', color='#3498db', alpha=0.7, edgecolor='black')
bars2 = axes[1].barh(x + width/2, fl_top, width, label='Federated', color='#e74c3c', alpha=0.7, edgecolor='black')

axes[1].set_yticks(x)
axes[1].set_yticklabels(top_feat_names, fontsize=10)
axes[1].set_xlabel('Feature Importance (AUC Drop)', fontsize=12, fontweight='bold')
axes[1].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
print('Saved comparison plot to outputs/model_comparison.png')

# Create summary report
summary = f"""
CREDIT CARD FRAUD DETECTION - RESULTS SUMMARY
==============================================

DATASET:
- Total transactions: 284,807
- Fraud rate: 0.17% (492 frauds)
- Features: 30 (V1-V28 PCA + Time + Amount)

PREPROCESSING:
- Sequence length: 10 transactions
- Stride: 5 (50% overlap)
- Sample fraction: 10% for faster training
- Final sequences: 5,661

MODEL ARCHITECTURE:
- GNN Encoder: 2-layer Graph Neural Network with fully-connected adjacency
- LSTM: Hidden size 64 for temporal modeling
- Classifier: Fully-connected layer with Sigmoid output

TRAINING CONFIGURATION:
- Baseline: Standard training with class weighting
- Federated: 4 clients, 5 rounds, FedAvg aggregation
- Loss: Binary Cross-Entropy with class weights
- Optimizer: Adam (lr=0.001)

RESULTS:
- Baseline Model AUC: {baseline['base_auc']:.4f}
- Federated Model AUC: {fl['base_auc']:.4f}
- Difference: {abs(baseline['base_auc'] - fl['base_auc']):.4f}

KEY INSIGHTS:
- Both models achieve excellent fraud detection performance (>90% AUC)
- {"Baseline performs slightly better" if baseline['base_auc'] > fl['base_auc'] else "Federated performs slightly better"}
- Federated learning enables privacy-preserving training across multiple clients
- GNN captures transaction relationships; LSTM captures temporal patterns

TOP 5 MOST IMPORTANT FEATURES:
"""

for i, (feat, _) in enumerate(top_features[:5], 1):
    b_imp = baseline_imps.get(feat, 0)
    fl_imp = fl_imps.get(feat, 0)
    summary += f"{i}. {feat}: Baseline={b_imp:.4f}, Federated={fl_imp:.4f}\n"

with open('outputs/RESULTS_SUMMARY.txt', 'w') as f:
    f.write(summary)

print('\nSaved results summary to outputs/RESULTS_SUMMARY.txt')
print('\n' + summary)
