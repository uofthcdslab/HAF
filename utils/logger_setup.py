import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_level='INFO', log_dir=None):
    """
    Set up and configure a logger
    
    Args:
        name (str): Name of the logger
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir (str, optional): Directory to store log files. If None, logs to current directory
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if log_dir:
        log_path = Path(f"logs/{log_dir}")
    else:
        log_path = Path(f"logs/")
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = f"{log_path}/{name}_{timestamp}.log"
        
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized with level {log_level}")
    return logger 