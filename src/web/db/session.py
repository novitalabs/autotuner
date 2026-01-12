"""
Database session management.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
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
		"timeout": 30,  # 30-second timeout for lock acquisition
	},
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
	engine,
	class_=AsyncSession,
	expire_on_commit=False,
)


async def init_db():
	"""Initialize database (create tables) and enable WAL mode."""
	async with engine.begin() as conn:
		# Enable WAL (Write-Ahead Logging) mode for concurrent writes
		# WAL mode allows multiple readers and one writer to access the database simultaneously
		# This is critical for parallel experiment execution
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
