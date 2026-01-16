import logging
import sys

def setup_logging():
    # Create a logger
    logger = logging.getLogger('crsm')
    logger.setLevel(logging.INFO)

    # Create a console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(ch)

    return logger

logger = setup_logging()
