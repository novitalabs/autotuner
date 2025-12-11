"""
Development server runner.
"""

import sys
from pathlib import Path
import uvicorn

if __name__ == "__main__":
	# Ensure we're running from src/ directory
	src_dir = Path(__file__).parent.parent
	sys.path.insert(0, str(src_dir))

	# Import settings after path is configured
	from web.config import get_settings
	settings = get_settings()

	uvicorn.run(
		"web.app:app",
		host=settings.server_host,
		port=settings.server_port,
		reload=True,  # Enable hot reload for development
		reload_dirs=[str(src_dir)],  # Watch src/ directory
		log_level="info",
	)
