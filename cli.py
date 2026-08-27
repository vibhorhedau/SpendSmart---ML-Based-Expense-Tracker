import sqlite3
import datetime
import os
import csv
from migrate_db import migrate_db
from categorizer_ml import predict
from retrain import run_retraining_pipeline

DB_FILE = 'expenses.db'
FEEDBACK_CSV = 'user_feedback_data.csv'
COMBINED_CSV = 'combined_training_data.csv'

def connect_db():
    """Connects to the SQLite database."""
    return sqlite3.connect(DB_FILE)

def init_db():
    """Ensures the table exists and columns are up to date."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expense (
            id INTEGER PRIMARY KEY,
            title VARCHAR(100) NOT NULL,
            amount FLOAT NOT NULL,
            category VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            description TEXT,
            predicted_category VARCHAR(50),
            confidence_score FLOAT,
            is_user_corrected INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()
    
    # Run migration script to ensure existing DB files get updated seamlessly
    migrate_db(DB_FILE)

def log_user_override(title, amount, category, date_str, description, predicted_category, confidence_score):
    """
    Logs user-corrected or custom categories to user_feedback_data.csv and combined_training_data.csv.
    This creates an active feedback loop for continuous model improvement.
    """
    headers = [
        'id', 'title', 'amount', 'category', 'date', 'description',
        'predicted_category', 'confidence_score', 'is_user_corrected', 'source'
    ]
    
    timestamp_id = f"user_override_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    row_dict = {
        'id': timestamp_id,
        'title': title,
        'amount': str(amount),
        'category': category,
        'date': date_str,
        'description': description if description else f"Predicted: {predicted_category}",
        'predicted_category': predicted_category,
        'confidence_score': str(confidence_score),
        'is_user_corrected': '1',
        'source': 'user_override_feedback'
    }

    # 1. Append to dedicated feedback CSV
    file_exists = os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

    # 2. Append to combined training dataset
    if os.path.exists(COMBINED_CSV):
        with open(COMBINED_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row_dict)

    print(f"🔄 [Feedback Loop Logged] Saved user override ('{title}' ➔ '{category}') to training dataset!")

def add_expense():
    print("\n--- Add Expense ---")
    title = input("Title: ").strip()
    if not title:
        print("Title cannot be empty.")
        return

    print("Select Currency: [1] USD ($)  [2] INR (₹)")
    curr_input = input("Enter choice (1 or 2) [default: 1]: ").strip()
    
    try:
        if curr_input == "2":
            amount_inr = float(input("Amount (in ₹ INR): "))
            amount = amount_inr / 83.0
            print(f"-> Converted: ₹{amount_inr:,.2f} INR ≈ ${amount:,.2f} USD")
        else:
            amount = float(input("Amount (in $ USD): "))
    except ValueError:
        print("Invalid amount. Please enter a number.")
        return
    
    description = input("Description (optional): ").strip()
    
    # Run ML Model v1 auto-prediction (with confidence fallback to rule engine)
    full_text = f"{title} {description}".strip()
    predicted_cat, conf, src = predict(full_text, amount)
    
    src_label = "🤖 ML Model v1" if src == "ml_model" else "⚡ Rule Engine Fallback"
    
    print("\nAvailable Categories: Food, Transport, Entertainment, Utilities, Shopping, Other")
    print(f"{src_label} Suggested Category: '{predicted_cat}' (Confidence: {conf:.2f})")
    
    user_input = input(f"Accept '{predicted_cat}'? [Press Enter/Y to accept, or type custom category]: ").strip()
    
    if user_input.lower() in ['', 'y', 'yes', predicted_cat.lower()]:
        category = predicted_cat
        is_user_corrected = 0
    else:
        category = user_input
        is_user_corrected = 1

    date_str = input("Date (YYYY-MM-DD) [leave empty for today]: ").strip()
    if not date_str:
        date_str = datetime.date.today().strftime("%Y-%m-%d")

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO expense (title, amount, category, date, description, predicted_category, confidence_score, is_user_corrected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (title, amount, category, date_str, description, predicted_cat, conf, is_user_corrected))
    conn.commit()
    conn.close()

    print(f"\n✅ Expense added successfully! Saved Category: '{category}'")

    # If user corrected/overrode prediction, trigger active feedback logging
    if is_user_corrected == 1:
        log_user_override(title, amount, category, date_str, description, predicted_cat, conf)

def view_expenses():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, amount, category, date, description, predicted_category, confidence_score, is_user_corrected FROM expense ORDER BY date DESC')
    expenses = cursor.fetchall()
    conn.close()

    print("\n--- All Expenses ---")
    print(f"{'ID':<5} {'Date':<12} {'Category':<15} {'Amount':<10} {'Title'}")
    print("-" * 78)
    
    if not expenses:
        print("No expenses found.")
    
    for exp in expenses:
        eid, title, amount, category, date, description, pred_cat, conf, is_corr = exp
        extra = ""
        if pred_cat:
            corr_flag = " [User Corrected]" if is_corr else ""
            extra = f" (Predicted: {pred_cat}, Conf: {conf:.2f}{corr_flag})"
        print(f"{eid:<5} {date:<12} {category:<15} ${amount:<9.2f} {title}{extra}")

def delete_expense():
    view_expenses()
    try:
        eid = int(input("\nEnter ID of expense to delete: "))
    except ValueError:
        print("Invalid ID.")
        return

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM expense WHERE id = ?', (eid,))
    if cursor.rowcount > 0:
        print("Expense deleted.")
    else:
        print("Expense not found.")
    conn.commit()
    conn.close()

def retrain_model_cli():
    """
    CLI action to trigger ML model re-training and update lifecycle history.
    """
    print("\n--- Re-training ML Model & Updating Lifecycle History ---")
    try:
        run_retraining_pipeline()
        print("Model re-trained, lifecycle history logged, and growth chart updated successfully!")
    except Exception as e:
        print(f"Error re-training model: {e}")

def main():
    init_db()
    while True:
        print("\n=== SpendSmart ML Expense Tracker CLI ===")
        print("1. View Expenses")
        print("2. Add Expense (ML Auto-Categorization)")
        print("3. Delete Expense")
        print("4. Re-train ML Model & Update Lifecycle History")
        print("5. Exit")
        
        choice = input("Select an option: ").strip()
        
        if choice == '1':
            view_expenses()
        elif choice == '2':
            add_expense()
        elif choice == '3':
            delete_expense()
        elif choice == '4':
            retrain_model_cli()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
