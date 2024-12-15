import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(
    name,
    log_dir="logs",
    log_file=None,
    level=logging.INFO,
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
    debug=False
):
    """
    Set up a logger with a rotating file handler and a stream handler.

    Parameters:
        name (str): Name of the logger.
        log_dir (str): Directory where log files will be saved.
        log_file (str): Name of the log file (default: `name.log`).
        level (int): Logging level (default: logging.INFO).
        max_bytes (int): Maximum size of the log file before rotation (default: 5MB).
        backup_count (int): Number of backup log files to keep (default: 3).
        debug (bool): Enable debug mode to override log level to logging.DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    level = logging.DEBUG if debug else level

    if log_file is None:
        log_file = f"{name}.log"

    log_path = os.path.join(log_dir, log_file)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        # Rotating File Handler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream Handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_formatter = logging.Formatter("%(levelname)s: %(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger


# Example usage
if __name__ == "__main__":
    logger = setup_logger("AppLogger", debug=True)
    if logger:
        logger.debug("This is a debug message.")
        logger.info("Application started.")
        logger.warning("This is a warning.")
        logger.error("An error occurred.")
        logger.critical("Critical issue detected!")
