import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class VerboseFilter(logging.Filter):
    """Filter out records marked with the verbose attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (hasattr(record, "verbose") and record.verbose)


def _file_handler(path: Path, *, max_bytes: int, backup_count: int) -> logging.Handler:
    if max_bytes <= 0:
        return logging.FileHandler(path, encoding="utf-8")
    return RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=max(0, backup_count),
        encoding="utf-8",
    )


def setup_logging(cfg: Any) -> logging.Logger:
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging_cfg = getattr(cfg, "logging", None)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format=log_format,
        handlers=[],
        force=True,
    )
    if bool(getattr(logging_cfg, "suppress_httpx_logs", True)):
        logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger("MLEvolve")
    logger.handlers.clear()
    logger.propagate = False
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    backup_count = int(getattr(logging_cfg, "log_backup_count", 2) or 0)

    if bool(getattr(logging_cfg, "write_brief_log", True)):
        brief_name = str(getattr(logging_cfg, "brief_log_filename", "MLEvolve.log"))
        file_handler = _file_handler(
            cfg.log_dir / brief_name,
            max_bytes=int(getattr(logging_cfg, "brief_log_max_bytes", 67108864) or 0),
            backup_count=backup_count,
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.addFilter(VerboseFilter())
        logger.addHandler(file_handler)

    if bool(getattr(logging_cfg, "write_verbose_log", True)):
        verbose_name = str(getattr(logging_cfg, "verbose_log_filename", "MLEvolve.verbose.log"))
        verbose_file_handler = _file_handler(
            cfg.log_dir / verbose_name,
            max_bytes=int(getattr(logging_cfg, "verbose_log_max_bytes", 268435456) or 0),
            backup_count=backup_count,
        )
        verbose_file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(verbose_file_handler)

    if bool(getattr(logging_cfg, "write_console_log", True)):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.addFilter(VerboseFilter())
        logger.addHandler(console_handler)
    return logger
