"""
Database migration: Add quant_config column to tasks table.

This migration adds the `quant_config` JSON column to store
runtime quantization configuration (gemm_dtype, kvcache_dtype, attention_dtype, moe_dtype).
"""

import sqlite3
import sys
from pathlib import Path

# Database location (XDG-compliant)
DB_PATH = Path.home() / ".local/share/inference-autotuner/autotuner.db"


def migrate_add_quant_config():
    """Add quant_config column to tasks table."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("No migration needed (database will be created with new schema)")
        return

    print(f"Migrating database: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [row[1] for row in cursor.fetchall()]

        if "quant_config" in columns:
            print("✓ Column 'quant_config' already exists, skipping migration")
            return

        # Add the new column
        print("Adding column 'quant_config' to tasks table...")
        cursor.execute("""
            ALTER TABLE tasks
            ADD COLUMN quant_config JSON
        """)

        conn.commit()
        print("✓ Migration completed successfully!")

        # Verify the column was added
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "quant_config" in columns, "Migration failed: column not found"

        print(f"✓ Verified: tasks table now has {len(columns)} columns")

    except Exception as e:
        print(f"✗ Migration failed: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    migrate_add_quant_config()
