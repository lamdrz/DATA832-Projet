import logging
from datetime import datetime
from pathlib import Path

from config.settings import DEBUG


def setup_logging(base_dir, debug=False):
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    resolved_base_dir = Path(base_dir)
    logs_folder_path = resolved_base_dir / "logs"
    logs_folder_path.mkdir(parents=True, exist_ok=True)

    log_file_path = logs_folder_path / f"{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.DEBUG if (debug if debug else DEBUG) else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
        handlers=[
            logging.FileHandler(log_file_path, "a", "utf-8"),
            logging.StreamHandler(),
        ],
    )


def get_logger(name) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
