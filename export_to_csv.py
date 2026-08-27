import sqlite3
import csv
import os
import sys

DB_FILE = 'expenses.db'
DEFAULT_OUTPUT_CSV = 'spendsmart_export.csv'

def export_expenses_to_csv(db_path=DB_FILE, output_csv=DEFAULT_OUTPUT_CSV):
    """
    Exports all expense records from the SQLite database to a CSV file.
    Serves as the initial user-provided dataset for training ML models.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found. Please run cli.py or add expenses first.")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT id, title, amount, category, date, description, 
                   predicted_category, confidence_score, is_user_corrected 
            FROM expense ORDER BY id ASC
        ''')
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Error querying table 'expense': {e}")
        conn.close()
        return False

    conn.close()

    headers = [
        'id', 'title', 'amount', 'category', 'date', 'description',
        'predicted_category', 'confidence_score', 'is_user_corrected', 'source'
    ]

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        
        for r in rows:
            # Append 'spendsmart' as source indicator
            writer.writerow(list(r) + ['spendsmart'])

    print(f"Successfully exported {len(rows)} entries from '{db_path}' to '{output_csv}'.")
    return True

if __name__ == '__main__':
    output_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_CSV
    export_expenses_to_csv(output_csv=output_path)
