import os
import csv
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_FILE = 'model_v1.joblib'
COMBINED_CSV = 'combined_training_data.csv'
FEEDBACK_CSV = 'user_feedback_data.csv'
HISTORY_JSON = 'training_history.json'
HISTORY_CSV = 'training_history.csv'
PLOT_FILE = 'accuracy_over_time.png'

def load_all_training_data():
    """
    Loads and deduplicates all available training records from 
    combined_training_data.csv and user_feedback_data.csv.
    """
    records = []
    seen_keys = set()

    sources = [COMBINED_CSV, FEEDBACK_CSV]
    for src in sources:
        if os.path.exists(src):
            with open(src, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title = row.get('title', '').strip()
                    desc = row.get('description', '').strip()
                    cat = row.get('category', '').strip()

                    if not cat or not title:
                        continue

                    full_text = f"{title} {desc}".strip()
                    dedup_key = (full_text.lower(), cat.lower())

                    # Prefer user feedback overrides if available
                    if dedup_key not in seen_keys or src == FEEDBACK_CSV:
                        seen_keys.add(dedup_key)
                        records.append({
                            'full_text': full_text,
                            'category': cat,
                            'source': row.get('source', 'dataset')
                        })

    df = pd.DataFrame(records)
    return df

def update_history_log(entry):
    """
    Appends a new training run metric entry to training_history.json and training_history.csv.
    """
    history = []
    if os.path.exists(HISTORY_JSON):
        try:
            with open(HISTORY_JSON, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(entry)

    with open(HISTORY_JSON, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(HISTORY_CSV, index=False)
    return history

def generate_lifecycle_plot(history):
    """
    Generates and saves the ML Lifecycle Accuracy Growth chart (accuracy_over_time.png).
    Demonstrates model performance improvement as dataset size increases.
    """
    if not history or len(history) < 1:
        return

    df = pd.DataFrame(history)
    
    plt.figure(figsize=(10, 6))
    
    x = range(1, len(df) + 1)
    
    plt.plot(x, df['accuracy'], marker='o', linewidth=2.5, color='#1f77b4', label='Accuracy (%)')
    plt.plot(x, df['f1_score'], marker='s', linewidth=2.5, color='#2ca02c', linestyle='--', label='Macro F1-Score (%)')
    
    for i, txt in enumerate(df['accuracy']):
        plt.annotate(f"{txt:.1f}% (N={df['total_samples'].iloc[i]})", 
                     (x[i], df['accuracy'].iloc[i]),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

    plt.title('SpendSmart ML Model Lifecycle: Performance vs Data Growth', fontsize=13, fontweight='bold')
    plt.xlabel('Retraining Cycle Iteration', fontsize=11)
    plt.ylabel('Score (%)', fontsize=11)
    plt.xticks(x, [f"v{v}" for v in df['version']])
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()
    print(f"📈 [ML Lifecycle] Updated accuracy growth chart saved to '{PLOT_FILE}'.")

def run_retraining_pipeline():
    """
    Executes full retraining pipeline:
    1. Ingests all baseline + user feedback data.
    2. Fits Logistic Regression classification pipeline.
    3. Evaluates test set metrics.
    4. Updates model_v1.joblib.
    5. Records lifecycle history and updates progression plot.
    """
    print("--- Executing SpendSmart ML Retraining Pipeline ---")
    df = load_all_training_data()
    
    if len(df) < 10:
        print("Not enough training samples to perform retraining.")
        return

    X = np.array(df['full_text'].tolist())
    y = np.array(df['category'].tolist())
    feedback_count = len(df[df['source'].astype(str).str.contains('user', case=False, na=False)])

    print(f"Total training corpus: {len(X)} samples ({feedback_count} user feedback overrides).")

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2500, lowercase=True, strip_accents='unicode')),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    prec, rec, f1 = round(prec * 100, 2), round(rec * 100, 2), round(f1 * 100, 2)

    # Save updated model pipeline
    joblib.dump(pipeline, MODEL_FILE)
    print(f"✅ Updated trained model pipeline saved to '{MODEL_FILE}'.")

    # Version timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    entry = {
        'timestamp': timestamp,
        'version': version_str,
        'total_samples': len(X),
        'user_feedback_count': feedback_count,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

    history = update_history_log(entry)
    generate_lifecycle_plot(history)

    print(f"\n========================================================")
    print(f"🔄 Retraining Pipeline Completed Successfully")
    print(f"========================================================")
    print(f"Version            : {version_str}")
    print(f"Total Dataset Size : {len(X)} samples")
    print(f"User Overrides     : {feedback_count} feedback samples")
    print(f"Retrained Accuracy : {acc:.2f}%")
    print(f"Retrained F1-Score : {f1:.2f}%")
    print(f"========================================================\n")
    return entry

if __name__ == '__main__':
    run_retraining_pipeline()
