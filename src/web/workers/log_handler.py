"""
Custom logging handler for remote workers.

Captures log messages and publishes them to Redis for real-time streaming
to the manager and frontend.
"""

import logging
import asyncio
from typing import Optional
from queue import Queue
from threading import Thread


class AsyncRedisLogHandler(logging.Handler):
    """Logging handler that publishes log messages to Redis.

    Uses a background thread to avoid blocking the main event loop.
    """

    def __init__(
        self,
        worker_id: str,
        level: int = logging.INFO,
        task_id: Optional[int] = None,
    ):
        super().__init__(level)
        self.worker_id = worker_id
        self.task_id = task_id
        self._queue: Queue = Queue()
        self._running = False
        self._thread: Optional[Thread] = None
        self._publisher = None

        # Format: [timestamp] [level] message
        self.setFormatter(logging.Formatter('%(message)s'))

    def start(self):
        """Start the background publishing thread."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._run_publisher, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background publishing thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def set_task_context(self, task_id: Optional[int], experiment_id: Optional[int] = None):
        """Set current task/experiment context for log entries."""
        self.task_id = task_id
        self.experiment_id = experiment_id

    def emit(self, record: logging.LogRecord):
        """Queue a log record for publishing."""
        if not self._running:
            return

        try:
            # Format the message
            msg = self.format(record)

            # Determine source from logger name
            source = "worker"
            if "orchestrator" in record.name.lower():
                source = "task"
            elif "benchmark" in record.name.lower() or "genai" in record.name.lower():
                source = "benchmark"
            elif "controller" in record.name.lower():
                source = "deployment"

            entry = {
                "message": msg,
                "level": record.levelname,
                "source": source,
                "task_id": getattr(self, 'task_id', None),
                "experiment_id": getattr(self, 'experiment_id', None),
            }
            self._queue.put(entry)
        except Exception:
            self.handleError(record)

    def _run_publisher(self):
        """Background thread that publishes queued logs."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._publish_loop())
        finally:
            loop.close()

    async def _publish_loop(self):
        """Async loop that processes the log queue."""
        try:
            # Try both import paths since the PYTHONPATH might vary
            try:
                from src.web.workers.pubsub import get_result_publisher
            except ImportError:
                from web.workers.pubsub import get_result_publisher
        except ImportError as e:
            logging.getLogger(__name__).error(f"Failed to import pubsub module: {e}")
            return

        try:
            self._publisher = await get_result_publisher(self.worker_id)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to create log publisher: {e}")
            return

        while self._running:
            try:
                # Process all queued items
                while not self._queue.empty():
                    try:
                        entry = self._queue.get_nowait()
                        await self._publisher.publish_log(
                            message=entry["message"],
                            level=entry["level"],
                            source=entry["source"],
                            task_id=entry["task_id"],
                            experiment_id=entry["experiment_id"],
                        )
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Failed to publish log: {e}")

                # Sleep briefly between checks
                await asyncio.sleep(0.1)

            except Exception as e:
                logging.getLogger(__name__).error(f"Error in log publish loop: {e}")
                await asyncio.sleep(1)


# Global handler instance
_log_handler: Optional[AsyncRedisLogHandler] = None


def setup_remote_logging(worker_id: str, level: int = logging.INFO) -> AsyncRedisLogHandler:
    """Setup remote logging for a worker.

    Args:
        worker_id: Worker identifier
        level: Minimum log level to publish

    Returns:
        The configured log handler
    """
    global _log_handler

    if _log_handler is not None:
        return _log_handler

    _log_handler = AsyncRedisLogHandler(worker_id, level)
    _log_handler.start()

    # Add to root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.addHandler(_log_handler)

    # IMPORTANT: Set root logger level to capture INFO and above
    # By default it's WARNING which filters out INFO messages
    if root_logger.level > level or root_logger.level == logging.NOTSET:
        root_logger.setLevel(level)

    # Also add to specific loggers we care about
    for name in ["arq", "src.orchestrator", "src.controllers", "web.workers"]:
        logger = logging.getLogger(name)
        if _log_handler not in logger.handlers:
            logger.addHandler(_log_handler)
        # Ensure these loggers also have appropriate level
        if logger.level > level or logger.level == logging.NOTSET:
            logger.setLevel(level)

    return _log_handler


def get_log_handler() -> Optional[AsyncRedisLogHandler]:
    """Get the global log handler."""
    return _log_handler


def shutdown_remote_logging():
    """Shutdown remote logging."""
    global _log_handler
    if _log_handler:
        _log_handler.stop()

        # Remove from loggers
        root_logger = logging.getLogger()
        if _log_handler in root_logger.handlers:
            root_logger.removeHandler(_log_handler)

        _log_handler = None
