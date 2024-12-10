import logging

def setup_logger(log_file="results/logs/training.log"):
    """
    Set up a logger to log training details to a file and console.

    Args:
        log_file (str): Path to the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
