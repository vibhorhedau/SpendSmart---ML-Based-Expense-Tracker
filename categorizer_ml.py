import os
import csv
import sys
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from categorizer_rules import predict_category as rule_predict

MODEL_FILE = 'model_v1.joblib'
DEFAULT_DATASET = 'combined_training_data.csv'
CONFIDENCE_THRESHOLD = 0.50

_global_model_pipeline = None

def load_dataset(dataset_path=DEFAULT_DATASET):
    """
    Loads labeled training dataset from CSV.
    Combines title and description for richer text features.
    """
    if not os.path.exists(dataset_path):
        if os.path.exists('processed_kaggle_dataset.csv'):
            dataset_path = 'processed_kaggle_dataset.csv'
        else:
            raise FileNotFoundError(f"Dataset '{dataset_path}' not found. Please run Phase 1 dataset scripts first.")

    texts = []
    labels = []

    with open(dataset_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('title', '').strip()
            desc = row.get('description', '').strip()
            category = row.get('category', '').strip()

            if not category or not title:
                continue

            full_text = f"{title} {desc}".strip()
            texts.append(full_text)
            labels.append(category)

    return texts, labels

def train_and_evaluate(dataset_path=DEFAULT_DATASET, model_save_path=MODEL_FILE):
    """
    Trains TF-IDF + Multinomial Naive Bayes pipeline on training dataset,
    evaluates metrics on an 80/20 test split, and saves the trained model to disk.
    """
    print(f"--- Training Phase 3 ML Model v1 (TF-IDF + Multinomial Naive Bayes) ---")
    texts, labels = load_dataset(dataset_path)
    
    print(f"Loaded {len(texts)} total labeled samples from '{dataset_path}'.")
    print(f"Class distribution: {dict(Counter(labels))}\n")

    # 80/20 train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build ML pipeline: TF-IDF vectorizer + Multinomial Naive Bayes classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            lowercase=True,
            strip_accents='unicode'
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])

    # Fit model on training set
    print("Fitting model on training split...")
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Calculate overall metrics
    acc = accuracy_score(y_test, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n========================================================")
    print(f"🎯 ML Model v1 Test Evaluation Metrics")
    print(f"========================================================")
    print(f"Test Set Accuracy   : {acc:.2f}%")
    print(f"Macro Precision     : {precision * 100:.2f}%")
    print(f"Macro Recall        : {recall * 100:.2f}%")
    print(f"Macro F1-Score      : {f1 * 100:.2f}%")
    print(f"========================================================\n")

    # Detailed per-class classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Detailed Category Report:\n", report)

    # Save model pipeline using joblib
    joblib.dump(pipeline, model_save_path)
    print(f"Trained model pipeline successfully saved to '{model_save_path}'.\n")

    global _global_model_pipeline
    _global_model_pipeline = pipeline
    return pipeline

def get_model(model_path=MODEL_FILE):
    """
    Returns the loaded model pipeline, loading from disk or training if missing.
    """
    global _global_model_pipeline
    if _global_model_pipeline is not None:
        return _global_model_pipeline

    if os.path.exists(model_path):
        _global_model_pipeline = joblib.load(model_path)
        return _global_model_pipeline
    else:
        print(f"Model file '{model_path}' not found. Initializing training...")
        return train_and_evaluate(model_save_path=model_path)

def predict(text, amount=None, threshold=CONFIDENCE_THRESHOLD):
    """
    Predicts category for input text using ML Model v1 with fallback logic.
    
    Fallback behavior:
    If max ML prediction probability < threshold, defers prediction to 
    Phase 2 rule-based categorizer.
    
    Returns:
        tuple: (category, confidence_score, source)
               source is 'ml_model' or 'rule_fallback'
    """
    if not text or not isinstance(text, str):
        return ("Other", 0.1, "rule_fallback")

    try:
        model = get_model()
        classes = model.classes_
        probas = model.predict_proba([text])[0]

        max_idx = probas.argmax()
        max_prob = float(probas[max_idx])
        ml_category = classes[max_idx]

        if max_prob >= threshold:
            return (ml_category, round(max_prob, 2), "ml_model")
        else:
            # Low confidence in ML prediction -> defer to Rule engine
            rule_cat, rule_conf = rule_predict(text, amount)
            return (rule_cat, round(rule_conf, 2), "rule_fallback")
    except Exception as e:
        # Fallback to rule engine on error
        rule_cat, rule_conf = rule_predict(text, amount)
        return (rule_cat, round(rule_conf, 2), "rule_fallback")

if __name__ == '__main__':
    # Train and evaluate if script executed directly
    train_and_evaluate()

    # Interactive sanity test predictions
    test_cases = [
        "Score each.",
        "Race mr.",
        "Swiggy lunch food order",
        "Uber trip taxi to airport",
        "Electricity bill monthly payment",
        "Netflix monthly streaming subscription",
        "Amazon purchase new sneakers",
        "Unseen weird payment description"
    ]
    print("--- Phase 3 ML Model Inference Test ---")
    for sample in test_cases:
        cat, conf, src = predict(sample)
        print(f"Text: '{sample}' -> Category: '{cat}' | Conf: {conf} | Source: {src}")
