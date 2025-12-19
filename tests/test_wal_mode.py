#!/usr/bin/env python3
"""Test script to verify WAL mode is enabled."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from web.db.session import init_db, engine
from sqlalchemy import text


async def test_wal_mode():
    """Test if WAL mode is properly enabled."""
    print("Testing WAL mode configuration...")

    # Initialize database (this should enable WAL mode)
    print("\n1. Calling init_db() to enable WAL mode...")
    await init_db()
    print("   ✓ init_db() completed")

    # Check if WAL mode is enabled
    print("\n2. Checking journal_mode...")
    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA journal_mode"))
        mode = result.scalar()
        print(f"   Journal mode: {mode}")

        if mode.lower() == "wal":
            print("   ✓ WAL mode is ENABLED")
        else:
            print(f"   ✗ WAL mode is NOT enabled (current mode: {mode})")
            return False

    # Check other WAL-related settings
    print("\n3. Checking WAL configuration...")
    async with engine.begin() as conn:
        # Check synchronous mode
        result = await conn.execute(text("PRAGMA synchronous"))
        sync_mode = result.scalar()
        print(f"   synchronous: {sync_mode}")

        # Check busy timeout
        result = await conn.execute(text("PRAGMA busy_timeout"))
        timeout = result.scalar()
        print(f"   busy_timeout: {timeout}ms")

    print("\n✓ WAL mode test completed successfully!")
    print("\nBenefits of WAL mode:")
    print("  • Multiple readers can access database concurrently")
    print("  • Writers don't block readers")
    print("  • Critical for parallel experiment execution")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_wal_mode())
    sys.exit(0 if success else 1)
