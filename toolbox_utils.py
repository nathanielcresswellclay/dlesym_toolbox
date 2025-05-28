import sys
import logging

def setup_logging(log_path):
    """Redirect stdout/stderr and configure logging to go to a specified log file."""
    
    # Open the log file in append mode
    log_file = open(log_path, "w")

    # Redirect print statements and uncaught exceptions
    sys.stdout = log_file
    sys.stderr = log_file

    # Set up logging to write to the same file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(log_file)]
    )

    logger = logging.getLogger(__name__)
    return log_file, logger