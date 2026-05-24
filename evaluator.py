"""
Model Evaluation - Danh gia hieu nang he thong
"""
import pandas as pd
import numpy as np
import time, os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_visp_dataset(filepath, max_samples=None):
    print(f"Loading dataset: {filepath}")
    df = pd.read_excel(filepath)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)
    print(f"  Loaded {len(df)} samples")
    return df


def create_evaluation_dataset(df, n_positive=500, n_negative=500, seed=42):
    np.random.seed(seed)
    n_positive = min(n_positive, len(df))
    n_negative = min(n_negative, len(df))
    data = []
    pos_idx = np.random.choice(len(df), n_positive, replace=False)
    for idx in pos_idx:
        row = df.iloc[idx]
        o, p = str(row['original_text']), str(row['paraphrase_text'])
        if o and p and o != 'nan' and p != 'nan':
            data.append((o, p, 1))
    neg_idx = np.random.choice(len(df), n_negative*2, replace=False)
    count = 0
    for i in range(0, len(neg_idx)-1, 2):
        if count >= n_negative: break
        o = str(df.iloc[neg_idx[i]]['original_text'])
        p = str(df.iloc[neg_idx[i+1]]['paraphrase_text'])
        if o and p and o != 'nan' and p != 'nan':
            data.append((o, p, 0)); count += 1
    np.random.shuffle(data)
    print(f"  Created {len(data)} eval samples (pos:{sum(1 for d in data if d[2]==1)}, neg:{sum(1 for d in data if d[2]==0)})")
    return data


def evaluate_model(detector, eval_data, method='tfidf', threshold=0.5, verbose=True):
    print(f"\nEvaluating ({method}) on {len(eval_data)} samples, threshold={threshold}")
    y_true, y_pred, scores = [], [], []
    start = time.time()
    for i, (t1, t2, label) in enumerate(eval_data):
        result = detector.detect(t1, t2, method=method)
        score = result['similarity_score']
        y_true.append(label); y_pred.append(1 if score >= threshold else 0); scores.append(score)
        if verbose and (i+1) % 100 == 0: print(f"  Progress: {i+1}/{len(eval_data)}")
    total_time = time.time() - start
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    results = {'method':method,'threshold':threshold,'accuracy':round(acc,4),'precision':round(prec,4),
               'recall':round(rec,4),'f1_score':round(f1,4),'confusion_matrix':cm,
               'total_time':round(total_time,2),'scores':scores,'y_true':y_true,'y_pred':y_pred}
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Time: {total_time:.2f}s")
    return results


def plot_confusion_matrix(cm, method, save_path=None):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Different','Plagiarism'], yticklabels=['Different','Plagiarism'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {method.upper()}')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_score_distribution(results, save_path=None):
    scores, y_true = results['scores'], results['y_true']
    pos = [s for s,y in zip(scores,y_true) if y==1]
    neg = [s for s,y in zip(scores,y_true) if y==0]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(pos, bins=30, alpha=0.6, color='#ff6b6b', label='Paraphrase (Positive)', edgecolor='white')
    ax.hist(neg, bins=30, alpha=0.6, color='#4ecdc4', label='Different (Negative)', edgecolor='white')
    ax.axvline(x=results['threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold={results["threshold"]}')
    ax.set_xlabel('Similarity Score'); ax.set_ylabel('Count')
    ax.set_title(f'Score Distribution - {results["method"].upper()}')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_comparison(results_list, save_path=None):
    methods = [r['method'] for r in results_list]
    metrics = ['accuracy','precision','recall','f1_score']
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(methods)); width = 0.2
    colors = ['#667eea','#764ba2','#f093fb','#4facfe']
    for i, m in enumerate(metrics):
        vals = [r[m] for r in results_list]
        bars = ax.bar(x+i*width, vals, width, label=m.replace('_',' ').title(), color=colors[i])
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Method'); ax.set_ylabel('Score')
    ax.set_title('Method Comparison'); ax.set_xticks(x+width*1.5)
    ax.set_xticklabels([m.upper() for m in methods]); ax.legend(); ax.set_ylim(0,1.15)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig
