#!/usr/bin/env python3
"""
Database reset tool for LLM Autotuner.

This script provides utilities to reset the database:
- Drop all tables and recreate them
- Delete the database file completely
- Reset specific task status

Usage:
    python scripts/reset_db.py --drop-tables    # Drop and recreate all tables
    python scripts/reset_db.py --delete-db      # Delete database file completely
    python scripts/reset_db.py --reset-tasks    # Reset all tasks to PENDING
    python scripts/reset_db.py --task-id 1      # Reset specific task to PENDING
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from web.db.session import engine, AsyncSessionLocal
from web.db.models import Base, Task, Experiment, TaskStatus
from web.config import get_settings
from sqlalchemy import select
import asyncio


def get_database_path() -> Path:
    """Get the database file path."""
    settings = get_settings()
    # Extract path from database URL (sqlite+aiosqlite:///path/to/db)
    db_url = settings.database_url
    if "sqlite" in db_url:
        # Remove sqlite+aiosqlite:/// prefix
        db_path = db_url.split("///")[-1]
        return Path(db_path)
    else:
        raise ValueError(f"Only SQLite databases are supported. Got: {db_url}")


async def drop_and_recreate_tables():
    """Drop all tables and recreate them."""
    print("🔄 Dropping all tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("✅ All tables dropped")

    print("🔄 Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created successfully")


def delete_database_file():
    """Delete the database file completely."""
    db_path = get_database_path()

    if db_path.exists():
        print(f"🔄 Deleting database file: {db_path}")
        db_path.unlink()
        print("✅ Database file deleted successfully")
        print("💡 The database will be recreated when the server starts")
    else:
        print(f"ℹ️  Database file does not exist: {db_path}")


async def reset_all_tasks():
    """Reset all tasks to PENDING status."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Task))
        tasks = result.scalars().all()

        if not tasks:
            print("ℹ️  No tasks found in database")
            return

        print(f"🔄 Resetting {len(tasks)} task(s) to PENDING status...")
        for task in tasks:
            old_status = task.status
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.completed_at = None
            print(f"  • Task #{task.id} '{task.task_name}': {old_status} → PENDING")

        await session.commit()
        print("✅ All tasks reset to PENDING")


async def reset_specific_task(task_id: int):
    """Reset a specific task to PENDING status."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()

        if not task:
            print(f"❌ Task #{task_id} not found")
            return

        old_status = task.status
        task.status = TaskStatus.PENDING
        task.started_at = None
        task.completed_at = None
        await session.commit()

        print(f"✅ Task #{task_id} '{task.task_name}': {old_status} → PENDING")


async def list_tasks():
    """List all tasks in the database."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Task).order_by(Task.id))
        tasks = result.scalars().all()

        if not tasks:
            print("ℹ️  No tasks found in database")
            return

        print(f"\n📋 Found {len(tasks)} task(s):\n")
        for task in tasks:
            print(f"  ID: {task.id}")
            print(f"  Name: {task.task_name}")
            print(f"  Status: {task.status}")
            print(f"  Created: {task.created_at}")
            if task.started_at:
                print(f"  Started: {task.started_at}")
            if task.completed_at:
                print(f"  Completed: {task.completed_at}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Database reset tool for LLM Autotuner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Drop and recreate all tables (preserves database file)
  python scripts/reset_db.py --drop-tables

  # Delete database file completely
  python scripts/reset_db.py --delete-db

  # Reset all tasks to PENDING status
  python scripts/reset_db.py --reset-tasks

  # Reset specific task to PENDING status
  python scripts/reset_db.py --task-id 1

  # List all tasks
  python scripts/reset_db.py --list-tasks
        """
    )

    parser.add_argument(
        "--drop-tables",
        action="store_true",
        help="Drop all tables and recreate them (clears all data)"
    )

    parser.add_argument(
        "--delete-db",
        action="store_true",
        help="Delete the database file completely"
    )

    parser.add_argument(
        "--reset-tasks",
        action="store_true",
        help="Reset all tasks to PENDING status"
    )

    parser.add_argument(
        "--task-id",
        type=int,
        metavar="ID",
        help="Reset specific task to PENDING status"
    )

    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all tasks in the database"
    )

    args = parser.parse_args()

    # Ensure at least one action is specified
    if not any([args.drop_tables, args.delete_db, args.reset_tasks, args.task_id, args.list_tasks]):
        parser.print_help()
        sys.exit(1)

    try:
        if args.drop_tables:
            asyncio.run(drop_and_recreate_tables())

        if args.delete_db:
            delete_database_file()

        if args.reset_tasks:
            asyncio.run(reset_all_tasks())

        if args.task_id:
            asyncio.run(reset_specific_task(args.task_id))

        if args.list_tasks:
            asyncio.run(list_tasks())

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
