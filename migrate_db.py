import sqlite3
import os

DB_FILE = 'expenses.db'

def migrate_db(db_path=DB_FILE):
    """
    Migrates the SQLite database schema safely.
    Adds predicted_category, confidence_score, and is_user_corrected columns
    if they do not already exist in the expense table.
    """
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' does not exist yet. Migration will occur upon creation.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get existing columns in the 'expense' table
    cursor.execute("PRAGMA table_info(expense)")
    columns_info = cursor.fetchall()
    existing_columns = [col[1] for col in columns_info]

    columns_to_add = [
        ("predicted_category", "VARCHAR(50)"),
        ("confidence_score", "FLOAT"),
        ("is_user_corrected", "INTEGER DEFAULT 0")
    ]

    migrated = False
    for col_name, col_type in columns_to_add:
        if col_name not in existing_columns:
            alter_query = f"ALTER TABLE expense ADD COLUMN {col_name} {col_type};"
            print(f"Adding column '{col_name}' to 'expense' table...")
            cursor.execute(alter_query)
            migrated = True

    conn.commit()
    conn.close()

    if migrated:
        print("Database schema migration completed successfully.")
    else:
        print("Database schema is already up to date.")

if __name__ == "__main__":
    migrate_db()
