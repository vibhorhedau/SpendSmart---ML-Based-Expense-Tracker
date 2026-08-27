import csv
import os
import sys

# Standard SpendSmart Categories
SPENDSMART_CATEGORIES = {'Food', 'Transport', 'Entertainment', 'Utilities', 'Shopping', 'Other'}

# Mapping rules from Kaggle Personal Finance Dataset to SpendSmart taxonomy
CATEGORY_MAPPING = {
    'Food & Drink': 'Food',
    'Travel': 'Transport',
    'Entertainment': 'Entertainment',
    'Utilities': 'Utilities',
    'Shopping': 'Shopping',
    'Rent': 'Utilities',
    'Health & Fitness': 'Other',
    'Other': 'Other',
    'Salary': 'Other',
    'Investment': 'Other'
}

DEFAULT_KAGGLE_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Personal_Finance_Dataset.csv')
DEFAULT_PROCESSED_CSV = 'processed_kaggle_dataset.csv'
DEFAULT_COMBINED_CSV = 'combined_training_data.csv'

def process_kaggle_dataset(kaggle_path=DEFAULT_KAGGLE_CSV, output_path=DEFAULT_PROCESSED_CSV):
    """
    Reads the Kaggle personal finance dataset CSV, filters expense transactions,
    maps external categories into SpendSmart taxonomy, and writes clean output CSV.
    """
    if not os.path.exists(kaggle_path):
        # Fallback search locations
        possible_paths = [
            kaggle_path,
            '../Personal_Finance_Dataset.csv',
            'Personal_Finance_Dataset.csv'
        ]
        found = False
        for p in possible_paths:
            if os.path.exists(p):
                kaggle_path = p
                found = True
                break
        if not found:
            print(f"Error: Could not find Kaggle dataset at '{kaggle_path}'.")
            return False

    print(f"Reading Kaggle dataset from '{kaggle_path}'...")
    
    rows_processed = 0
    rows_skipped = 0
    mapped_records = []

    with open(kaggle_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            # Check transaction type: filter for Expense rows
            tx_type = row.get('Type', '').strip()
            if tx_type and tx_type.lower() != 'expense':
                rows_skipped += 1
                continue

            orig_cat = row.get('Category', '').strip()
            mapped_cat = CATEGORY_MAPPING.get(orig_cat, 'Other')
            
            description = row.get('Transaction Description', '').strip()
            amount = row.get('Amount', '0.0').strip()
            date = row.get('Date', '').strip()

            mapped_records.append({
                'id': f"kaggle_{idx}",
                'title': description,
                'amount': amount,
                'category': mapped_cat,
                'date': date,
                'description': f"Original Category: {orig_cat}",
                'predicted_category': '',
                'confidence_score': '1.0',
                'is_user_corrected': '0',
                'source': 'kaggle_dataset'
            })
            rows_processed += 1

    headers = [
        'id', 'title', 'amount', 'category', 'date', 'description',
        'predicted_category', 'confidence_score', 'is_user_corrected', 'source'
    ]

    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(mapped_records)

    print(f"Successfully processed {rows_processed} expense rows (skipped {rows_skipped} non-expense rows).")
    print(f"Cleaned supplementary dataset saved to '{output_path}'.")
    return True

def merge_datasets(spendsmart_csv='spendsmart_export.csv', kaggle_processed_csv=DEFAULT_PROCESSED_CSV, combined_csv=DEFAULT_COMBINED_CSV):
    """
    Merges user's SpendSmart export CSV with the mapped Kaggle supplementary dataset CSV
    into a unified training set file.
    """
    all_rows = []
    headers = None

    if os.path.exists(spendsmart_csv):
        with open(spendsmart_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            for r in reader:
                all_rows.append(r)
        print(f"Loaded {len(all_rows)} rows from SpendSmart export '{spendsmart_csv}'.")

    if os.path.exists(kaggle_processed_csv):
        with open(kaggle_processed_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not headers:
                headers = reader.fieldnames
            count_kaggle = 0
            for r in reader:
                all_rows.append(r)
                count_kaggle += 1
        print(f"Loaded {count_kaggle} rows from mapped Kaggle dataset '{kaggle_processed_csv}'.")

    if not headers:
        headers = [
            'id', 'title', 'amount', 'category', 'date', 'description',
            'predicted_category', 'confidence_score', 'is_user_corrected', 'source'
        ]

    with open(combined_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Total combined dataset: {len(all_rows)} rows saved to '{combined_csv}'.")

if __name__ == '__main__':
    kaggle_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KAGGLE_CSV
    success = process_kaggle_dataset(kaggle_path=kaggle_file)
    if success:
        merge_datasets()
