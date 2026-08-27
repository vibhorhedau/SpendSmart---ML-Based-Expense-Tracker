import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

DEFAULT_DATASET = 'combined_training_data.csv'
OUTPUT_PLOT = 'confusion_matrices.png'
OUTPUT_RESULTS_CSV = 'model_comparison_results.csv'

def load_data(dataset_path=DEFAULT_DATASET):
    """
    Loads labeled training data from CSV and returns full text features and labels.
    """
    if not os.path.exists(dataset_path):
        if os.path.exists('processed_kaggle_dataset.csv'):
            dataset_path = 'processed_kaggle_dataset.csv'
        else:
            raise FileNotFoundError(f"Dataset '{dataset_path}' not found.")

    df = pd.read_csv(dataset_path)
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['full_text'] = (df['title'] + ' ' + df['description']).str.strip()
    
    df = df[df['category'].notna() & (df['full_text'] != '')]
    return np.array(df['full_text'].tolist()), np.array(df['category'].tolist())

def run_model_comparison(dataset_path=DEFAULT_DATASET):
    """
    Trains and compares 4 ML models on the same dataset split,
    plots confusion matrices, and exports comparison results.
    """
    print(f"--- Phase 5: Model Comparison & Evaluation ---")
    X, y = load_data(dataset_path)
    
    unique_labels = sorted(list(set(y)))
    print(f"Total dataset size: {len(X)} records across {len(unique_labels)} categories: {unique_labels}\n")

    # 80/20 train/test split with stratification for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(C=1.0, random_state=42),
        'k-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    confusion_matrices = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (name, clf) in enumerate(models.items()):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=2500,
                lowercase=True,
                strip_accents='unicode'
            )),
            ('clf', clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        confusion_matrices[name] = cm

        results.append({
            'Model': name,
            'Accuracy (%)': round(acc, 2),
            'Macro Precision (%)': round(prec * 100, 2),
            'Macro Recall (%)': round(rec * 100, 2),
            'Macro F1-Score (%)': round(f1 * 100, 2)
        })

        # Plot Confusion Matrix Heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            ax=axes[idx],
            cbar=False
        )
        axes[idx].set_title(f"{name}\nAccuracy: {acc:.2f}% | F1: {f1*100:.2f}%", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Category', fontsize=10)
        axes[idx].set_ylabel('Actual Category', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print(f"✅ Confusion Matrix Grid visualization saved to '{OUTPUT_PLOT}'.")

    # Export comparison results table
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"✅ Comparison metrics summary saved to '{OUTPUT_RESULTS_CSV}'.\n")

    print("==========================================================================")
    print("🏆 Phase 5 Machine Learning Model Comparison Results")
    print("==========================================================================")
    print(df_results.to_string(index=False))
    print("==========================================================================\n")
    return df_results

if __name__ == '__main__':
    run_model_comparison()
