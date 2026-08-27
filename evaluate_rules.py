import csv
import os
import sys
from collections import defaultdict, Counter
from categorizer_rules import predict_category, RULES

DEFAULT_DATASET = 'combined_training_data.csv'

def evaluate(dataset_path=DEFAULT_DATASET):
    """
    Evaluates rule-based categorizer accuracy, precision, recall, and coverage
    against a labeled CSV dataset.
    """
    if not os.path.exists(dataset_path):
        # Fallback to processed Kaggle dataset if combined dataset not created yet
        if os.path.exists('processed_kaggle_dataset.csv'):
            dataset_path = 'processed_kaggle_dataset.csv'
        else:
            print(f"Error: Dataset '{dataset_path}' not found. Please run Phase 1 dataset scripts first.")
            return

    print(f"--- Rule-Based Categorizer Accuracy Evaluation ---")
    print(f"Evaluating dataset: {dataset_path}\n")

    total_samples = 0
    correct_predictions = 0
    rule_matches = 0  # Predictions that triggered a rule (conf > 0.1)

    # Per-category metric counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    ground_truth_counts = Counter()
    pred_counts = Counter()

    categories = list(RULES.keys())

    with open(dataset_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('title', '')
            actual_cat = row.get('category', '').strip()
            
            if not actual_cat:
                continue

            total_samples += 1
            ground_truth_counts[actual_cat] += 1

            pred_cat, conf = predict_category(title)
            pred_counts[pred_cat] += 1

            if conf > 0.1:
                rule_matches += 1

            if pred_cat == actual_cat:
                correct_predictions += 1
                tp[actual_cat] += 1
            else:
                fp[pred_cat] += 1
                fn[actual_cat] += 1

    if total_samples == 0:
        print("No valid labeled records found for evaluation.")
        return

    overall_accuracy = (correct_predictions / total_samples) * 100
    coverage_rate = (rule_matches / total_samples) * 100

    print(f"Total Samples Evaluated: {total_samples}")
    print(f"Correct Predictions    : {correct_predictions}")
    print(f"Overall Baseline Accuracy: {overall_accuracy:.2f}%")
    print(f"Rule Match Coverage    : {coverage_rate:.2f}% ({rule_matches}/{total_samples} matched specific rule)\n")

    print(f"{'Category':<15} {'Ground Truth':<14} {'Predicted':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 77)

    for cat in sorted(categories):
        actual_n = ground_truth_counts[cat]
        pred_n = pred_counts[cat]
        
        t_pos = tp[cat]
        f_pos = fp[cat]
        f_neg = fn[cat]

        precision = (t_pos / (t_pos + f_pos)) if (t_pos + f_pos) > 0 else 0.0
        recall = (t_pos / (t_pos + f_neg)) if (t_pos + f_neg) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"{cat:<15} {actual_n:<14} {pred_n:<12} {precision * 100:<11.2f}% {recall * 100:<11.2f}% {f1 * 100:<11.2f}%")

    print("-" * 77)
    return {
        'total_samples': total_samples,
        'accuracy': overall_accuracy,
        'coverage': coverage_rate
    }

if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    evaluate(ds)
