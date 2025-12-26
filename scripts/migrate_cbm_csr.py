"""
Database migration script to add ClusterBaseModel and ClusterServingRuntime columns.

Run this script to update the database schema with new columns.
"""

import sqlite3
import sys
from pathlib import Path

# Database path
DB_PATH = Path.home() / ".local/share/autotuner/autotuner.db"

def migrate():
    """Add new columns to tasks table."""

    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)

    print(f"Migrating database: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(tasks)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    columns_to_add = [
        ("clusterbasemodel_config", "JSON"),
        ("clusterservingruntime_config", "JSON"),
        ("created_clusterbasemodel", "VARCHAR"),
        ("created_clusterservingruntime", "VARCHAR"),
    ]

    for col_name, col_type in columns_to_add:
        if col_name in existing_columns:
            print(f"  ✓ Column '{col_name}' already exists, skipping")
        else:
            try:
                cursor.execute(f"ALTER TABLE tasks ADD COLUMN {col_name} {col_type}")
                print(f"  ✓ Added column '{col_name}' ({col_type})")
            except sqlite3.OperationalError as e:
                print(f"  ✗ Failed to add column '{col_name}': {e}")

    conn.commit()
    conn.close()

    print("\n✅ Migration completed successfully!")
    print("\nYou can now restart the server to use the new columns.")

if __name__ == "__main__":
    migrate()
