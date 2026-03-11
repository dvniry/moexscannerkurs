"""Безопасный логгер — маскирует токены и ключи."""
import logging
import re

_SENSITIVE = re.compile(
    r'(token|api[_-]?key|password|secret|authorization)["\s:=]+([^\s"&,}]{4})([^\s"&,}]*)',
    re.IGNORECASE
)

class SanitizingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SENSITIVE.sub(
            lambda m: f'{m.group(1)}="{m.group(2)}****"',
            str(record.msg)
        )
        return True

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        ))
        logger.addHandler(handler)
    logger.addFilter(SanitizingFilter())
    logger.setLevel(logging.INFO)
    return logger
