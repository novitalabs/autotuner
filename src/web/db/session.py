"""
Database session management.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.engine.url import make_url
from pathlib import Path
from web.config import get_settings
from web.db.models import Base

settings = get_settings()

# Create async engine with WAL mode support for concurrent writes
# WAL (Write-Ahead Logging) allows multiple readers and a single writer concurrently
engine = create_async_engine(
	settings.database_url,
	echo=settings.debug,
	future=True,
	connect_args={
		"check_same_thread": False,  # Allow SQLite to work across threads
		"timeout": 30,                # 30-second timeout for lock acquisition
	}
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
	engine,
	class_=AsyncSession,
	expire_on_commit=False,
)


async def init_db():
    """Initialize database (create tables) and enable WAL mode."""

    # --- 1) Ensure parent directory exists (for SQLite only) ---
    url = make_url(str(engine.url))

    if url.get_backend_name().startswith("sqlite"):
        # url.database is the absolute file path for sqlite
        db_path = Path(url.database) if url.database else None

        if db_path:
            db_dir = db_path.parent
            db_dir.mkdir(parents=True, exist_ok=True)

    # --- 2) Proceed with DB init ---
    async with engine.begin() as conn:
        # Enable WAL mode for better concurrency
        await conn.execute(text("PRAGMA journal_mode=WAL"))

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
	"""Dependency to get database session."""
	async with AsyncSessionLocal() as session:
		try:
			yield session
		finally:
			await session.close()
