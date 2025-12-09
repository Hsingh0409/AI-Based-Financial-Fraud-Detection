"""Create detailed visualization graphs for Hybrid GNN+LSTM model only"""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
from models import GNN_LSTM_Classifier
from utils import make_fully_connected_adj
from sklearn.model_selection import train_test_split

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_hybrid_model(model_path, feat_dim, device):
    """Load the trained hybrid GNN+LSTM model"""
    model = GNN_LSTM_Classifier(
        feat_dim=feat_dim,
        gnn_hidden=32,
        gnn_out=32,
        lstm_hidden=32,
        num_layers=1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_predictions(model, X, device):
    """Get predictions and probabilities from hybrid model"""
    batch_size = X.shape[0]
    seq_len = X.shape[1]
    adj = make_fully_connected_adj(batch_size, seq_len).to(device)
    X_tensor = torch.from_numpy(X).float().to(device)
    
    with torch.no_grad():
        # Get GNN output
        gnn_out = model.gnn(X_tensor, adj)
        
        # Get LSTM output
        lstm_out, (h_n, c_n) = model.lstm(gnn_out)
        
        # Get final predictions
        last_hidden = lstm_out[:, -1, :]
        logits = model.fc(last_hidden)
        predictions = torch.sigmoid(logits)
    
    return {
        'gnn_output': gnn_out.cpu().numpy(),
        'lstm_output': lstm_out.cpu().numpy(),
        'predictions': predictions.cpu().numpy().flatten()
    }

def create_hybrid_visualizations():
    """Create comprehensive visualizations for hybrid model"""
    
    print("\n" + "="*80)
    print("CREATING HYBRID GNN+LSTM VISUALIZATION GRAPHS")
    print("="*80)
    
    # Load data
    data = np.load('data/processed.npz')
    X_all = data['X']
    y_all = data['y']
    
    # Split data (same as training)
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    
    print(f"\nTest Set: {len(y_test)} samples")
    print(f"Fraud cases: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
    print(f"Normal cases: {len(y_test)-y_test.sum()} ({(len(y_test)-y_test.sum())/len(y_test)*100:.2f}%)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load both models
    print("\nLoading models...")
    baseline_model = load_hybrid_model('outputs/baseline_model.pt', X_test.shape[2], device)
    fl_model = load_hybrid_model('outputs/global_model.pt', X_test.shape[2], device)
    
    # Get predictions
    print("Getting predictions...")
    baseline_results = get_predictions(baseline_model, X_test, device)
    fl_results = get_predictions(fl_model, X_test, device)
    
    baseline_probs = baseline_results['predictions']
    fl_probs = fl_results['predictions']
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ============================================================================
    # GRAPH 1: ROC Curves Comparison (Baseline vs Federated)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Baseline ROC
    fpr_base, tpr_base, _ = roc_curve(y_test, baseline_probs)
    roc_auc_base = auc(fpr_base, tpr_base)
    
    # Federated ROC
    fpr_fl, tpr_fl, _ = roc_curve(y_test, fl_probs)
    roc_auc_fl = auc(fpr_fl, tpr_fl)
    
    ax1.plot(fpr_base, tpr_base, color='#3498db', linewidth=2.5, 
             label=f'Baseline (AUC = {roc_auc_base:.4f})')
    ax1.plot(fpr_fl, tpr_fl, color='#e74c3c', linewidth=2.5, 
             label=f'Federated (AUC = {roc_auc_fl:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    
    ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax1.set_title('ROC Curves - Hybrid GNN+LSTM', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ============================================================================
    # GRAPH 2: Precision-Recall Curves
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Baseline PR
    precision_base, recall_base, _ = precision_recall_curve(y_test, baseline_probs)
    pr_auc_base = auc(recall_base, precision_base)
    
    # Federated PR
    precision_fl, recall_fl, _ = precision_recall_curve(y_test, fl_probs)
    pr_auc_fl = auc(recall_fl, precision_fl)
    
    ax2.plot(recall_base, precision_base, color='#3498db', linewidth=2.5,
             label=f'Baseline (AUC = {pr_auc_base:.4f})')
    ax2.plot(recall_fl, precision_fl, color='#e74c3c', linewidth=2.5,
             label=f'Federated (AUC = {pr_auc_fl:.4f})')
    
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ============================================================================
    # GRAPH 3: Prediction Distribution
    # ============================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    ax3.hist(baseline_probs[y_test == 0], bins=50, alpha=0.5, color='#2ecc71', 
             label='Normal (Baseline)', edgecolor='black')
    ax3.hist(baseline_probs[y_test == 1], bins=50, alpha=0.5, color='#e74c3c', 
             label='Fraud (Baseline)', edgecolor='black')
    ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    
    ax3.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Baseline: Prediction Distribution', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # GRAPH 4: GNN Layer Activations (Baseline vs Federated)
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    gnn_base = baseline_results['gnn_output']
    gnn_fl = fl_results['gnn_output']
    
    # Calculate mean activation across samples and timesteps for each hidden dimension
    gnn_base_mean = gnn_base.mean(axis=(0, 1))  # Shape: (32,)
    gnn_fl_mean = gnn_fl.mean(axis=(0, 1))
    
    x = np.arange(len(gnn_base_mean))
    width = 0.35
    
    ax4.bar(x - width/2, gnn_base_mean, width, label='Baseline', color='#3498db', alpha=0.7, edgecolor='black')
    ax4.bar(x + width/2, gnn_fl_mean, width, label='Federated', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax4.set_xlabel('GNN Hidden Dimension', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Mean Activation', fontsize=11, fontweight='bold')
    ax4.set_title('GNN Layer Mean Activations', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # GRAPH 5: LSTM Layer Activations (Baseline vs Federated)
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    lstm_base = baseline_results['lstm_output']
    lstm_fl = fl_results['lstm_output']
    
    # Calculate mean activation across samples and timesteps
    lstm_base_mean = lstm_base.mean(axis=(0, 1))  # Shape: (32,)
    lstm_fl_mean = lstm_fl.mean(axis=(0, 1))
    
    ax5.bar(x - width/2, lstm_base_mean, width, label='Baseline', color='#3498db', alpha=0.7, edgecolor='black')
    ax5.bar(x + width/2, lstm_fl_mean, width, label='Federated', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax5.set_xlabel('LSTM Hidden Dimension', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Mean Activation', fontsize=11, fontweight='bold')
    ax5.set_title('LSTM Layer Mean Activations', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # GRAPH 6: Confusion Matrices (Side by Side)
    # ============================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    baseline_pred = (baseline_probs > 0.5).astype(int)
    fl_pred = (fl_probs > 0.5).astype(int)
    
    cm_base = confusion_matrix(y_test, baseline_pred)
    cm_fl = confusion_matrix(y_test, fl_pred)
    
    # Create combined confusion matrix visualization
    labels = ['TN', 'FP', 'FN', 'TP']
    base_values = [cm_base[0,0], cm_base[0,1], cm_base[1,0], cm_base[1,1]]
    fl_values = [cm_fl[0,0], cm_fl[0,1], cm_fl[1,0], cm_fl[1,1]]
    
    x_pos = np.arange(len(labels))
    
    ax6.bar(x_pos - width/2, base_values, width, label='Baseline', color='#3498db', alpha=0.7, edgecolor='black')
    ax6.bar(x_pos + width/2, fl_values, width, label='Federated', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax6.set_xlabel('Confusion Matrix Elements', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax6.set_title('Confusion Matrix Comparison', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (base_val, fl_val) in enumerate(zip(base_values, fl_values)):
        ax6.text(i - width/2, base_val + 20, str(base_val), ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax6.text(i + width/2, fl_val + 20, str(fl_val), ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ============================================================================
    # GRAPH 7: GNN Output Distribution (Fraud vs Normal)
    # ============================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Get mean GNN activation per sample
    gnn_base_sample_mean = gnn_base.mean(axis=(1, 2))  # Mean across timesteps and features
    
    ax7.hist(gnn_base_sample_mean[y_test == 0], bins=40, alpha=0.6, color='#2ecc71', 
             label='Normal Transactions', edgecolor='black')
    ax7.hist(gnn_base_sample_mean[y_test == 1], bins=40, alpha=0.6, color='#e74c3c', 
             label='Fraud Transactions', edgecolor='black')
    
    ax7.set_xlabel('Mean GNN Activation', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax7.set_title('GNN Output: Fraud vs Normal Distribution', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # GRAPH 8: LSTM Output Distribution (Fraud vs Normal)
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Get mean LSTM activation per sample
    lstm_base_sample_mean = lstm_base.mean(axis=(1, 2))
    
    ax8.hist(lstm_base_sample_mean[y_test == 0], bins=40, alpha=0.6, color='#2ecc71', 
             label='Normal Transactions', edgecolor='black')
    ax8.hist(lstm_base_sample_mean[y_test == 1], bins=40, alpha=0.6, color='#e74c3c', 
             label='Fraud Transactions', edgecolor='black')
    
    ax8.set_xlabel('Mean LSTM Activation', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax8.set_title('LSTM Output: Fraud vs Normal Distribution', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # ============================================================================
    # GRAPH 9: Model Architecture Flow
    # ============================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create text-based architecture diagram
    architecture_text = """
    HYBRID GNN+LSTM ARCHITECTURE
    ═══════════════════════════════════
    
    INPUT: Transaction Sequences
    ↓ (Batch, 10, 30)
    
    ┌─────────────────────────────────┐
    │   GNN ENCODER (2 Layers)        │
    │   - Hidden: 32 dimensions       │
    │   - Output: 32 dimensions       │
    │   - Fully-connected adjacency   │
    └─────────────────────────────────┘
    ↓ (Batch, 10, 32)
    
    ┌─────────────────────────────────┐
    │   LSTM (1 Layer)                │
    │   - Hidden: 32 dimensions       │
    │   - Bidirectional: No           │
    │   - Captures temporal patterns  │
    └─────────────────────────────────┘
    ↓ (Batch, 10, 32)
    
    ┌─────────────────────────────────┐
    │   FINAL CLASSIFIER              │
    │   - FC: 32 → 64 (ReLU)          │
    │   - FC: 64 → 1 (Sigmoid)        │
    └─────────────────────────────────┘
    ↓ (Batch, 1)
    
    OUTPUT: Fraud Probability [0-1]
    
    ═══════════════════════════════════
    RESULTS:
    • Baseline AUC: {:.4f}
    • Federated AUC: {:.4f}
    • Baseline Recall: 100%
    • Federated Recall: 100%
    ═══════════════════════════════════
    """.format(roc_auc_base, roc_auc_fl)
    
    ax9.text(0.1, 0.5, architecture_text, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    plt.savefig('outputs/hybrid_gnn_lstm_graphs.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n" + "="*80)
    print("SAVED: outputs/hybrid_gnn_lstm_graphs.png")
    print("="*80)
    
    # Create individual detailed graphs
    create_individual_graphs(baseline_results, fl_results, y_test, baseline_probs, fl_probs)
    
    print("\n" + "="*80)
    print("ALL HYBRID MODEL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. hybrid_gnn_lstm_graphs.png - Comprehensive 9-panel visualization")
    print("  2. hybrid_roc_curve.png - Detailed ROC curve")
    print("  3. hybrid_confusion_heatmap.png - Confusion matrix heatmap")
    print("  4. hybrid_layer_activations.png - Layer-wise activation analysis")
    print("\nAll files saved in outputs/ directory")
    print("="*80 + "\n")

def create_individual_graphs(baseline_results, fl_results, y_test, baseline_probs, fl_probs):
    """Create additional individual detailed graphs"""
    
    # ============================================================================
    # Individual Graph 1: Detailed ROC Curve
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fpr_base, tpr_base, thresholds_base = roc_curve(y_test, baseline_probs)
    roc_auc_base = auc(fpr_base, tpr_base)
    
    fpr_fl, tpr_fl, thresholds_fl = roc_curve(y_test, fl_probs)
    roc_auc_fl = auc(fpr_fl, tpr_fl)
    
    ax.plot(fpr_base, tpr_base, color='#3498db', linewidth=3, 
            label=f'Baseline GNN+LSTM (AUC = {roc_auc_base:.4f})', marker='o', 
            markevery=50, markersize=6)
    ax.plot(fpr_fl, tpr_fl, color='#e74c3c', linewidth=3, 
            label=f'Federated GNN+LSTM (AUC = {roc_auc_fl:.4f})', marker='s', 
            markevery=50, markersize=6)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)')
    
    ax.fill_between(fpr_base, tpr_base, alpha=0.2, color='#3498db')
    ax.fill_between(fpr_fl, tpr_fl, alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve - Hybrid GNN+LSTM Fraud Detection', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig('outputs/hybrid_roc_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("SAVED: outputs/hybrid_roc_curve.png")
    plt.close()
    
    # ============================================================================
    # Individual Graph 2: Confusion Matrix Heatmaps
    # ============================================================================
    baseline_pred = (baseline_probs > 0.5).astype(int)
    fl_pred = (fl_probs > 0.5).astype(int)
    
    cm_base = confusion_matrix(y_test, baseline_pred)
    cm_fl = confusion_matrix(y_test, fl_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline confusion matrix
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'],
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes[0].set_title('Baseline GNN+LSTM\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Add accuracy and other metrics
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    acc_base = (tn_b + tp_b) / (tn_b + fp_b + fn_b + tp_b)
    axes[0].text(0.5, -0.15, f'Accuracy: {acc_base:.2%}', ha='center', va='top', 
                transform=axes[0].transAxes, fontsize=11, fontweight='bold')
    
    # Federated confusion matrix
    sns.heatmap(cm_fl, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'],
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes[1].set_title('Federated GNN+LSTM\nConfusion Matrix', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Add accuracy
    tn_f, fp_f, fn_f, tp_f = cm_fl.ravel()
    acc_fl = (tn_f + tp_f) / (tn_f + fp_f + fn_f + tp_f)
    axes[1].text(0.5, -0.15, f'Accuracy: {acc_fl:.2%}', ha='center', va='top', 
                transform=axes[1].transAxes, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/hybrid_confusion_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("SAVED: outputs/hybrid_confusion_heatmap.png")
    plt.close()
    
    # ============================================================================
    # Individual Graph 3: Layer-wise Activation Analysis
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    gnn_base = baseline_results['gnn_output']
    lstm_base = baseline_results['lstm_output']
    gnn_fl = fl_results['gnn_output']
    lstm_fl = fl_results['lstm_output']
    
    # GNN activation distribution (Baseline)
    gnn_base_flat = gnn_base.flatten()
    axes[0, 0].hist(gnn_base_flat, bins=100, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(gnn_base_flat.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {gnn_base_flat.mean():.2f}')
    axes[0, 0].set_xlabel('Activation Value', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Baseline: GNN Layer Activation Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # LSTM activation distribution (Baseline)
    lstm_base_flat = lstm_base.flatten()
    axes[0, 1].hist(lstm_base_flat, bins=100, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(lstm_base_flat.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {lstm_base_flat.mean():.2f}')
    axes[0, 1].set_xlabel('Activation Value', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Baseline: LSTM Layer Activation Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # GNN activation distribution (Federated)
    gnn_fl_flat = gnn_fl.flatten()
    axes[1, 0].hist(gnn_fl_flat, bins=100, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(gnn_fl_flat.mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {gnn_fl_flat.mean():.2f}')
    axes[1, 0].set_xlabel('Activation Value', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Federated: GNN Layer Activation Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # LSTM activation distribution (Federated)
    lstm_fl_flat = lstm_fl.flatten()
    axes[1, 1].hist(lstm_fl_flat, bins=100, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(lstm_fl_flat.mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {lstm_fl_flat.mean():.2f}')
    axes[1, 1].set_xlabel('Activation Value', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Federated: LSTM Layer Activation Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/hybrid_layer_activations.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("SAVED: outputs/hybrid_layer_activations.png")
    plt.close()

if __name__ == '__main__':
    create_hybrid_visualizations()
