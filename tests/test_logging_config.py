from __future__ import annotations

from logging.handlers import RotatingFileHandler
from types import SimpleNamespace

from utils.logging_config import setup_logging


def test_logging_uses_configured_rotating_file_limits(tmp_path) -> None:
    cfg = SimpleNamespace(
        log_level="INFO",
        log_dir=tmp_path,
        logging=SimpleNamespace(
            write_brief_log=True,
            write_verbose_log=True,
            write_console_log=False,
            brief_log_filename="brief.log",
            verbose_log_filename="verbose.log",
            brief_log_max_bytes=1024,
            verbose_log_max_bytes=4096,
            log_backup_count=3,
            suppress_httpx_logs=True,
        ),
    )

    logger = setup_logging(cfg)
    handlers = [handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)]

    assert [handler.maxBytes for handler in handlers] == [1024, 4096]
    assert all(handler.backupCount == 3 for handler in handlers)
    for handler in handlers:
        handler.close()
