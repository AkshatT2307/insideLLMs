"""
logger.py — Tees all output to both terminal and a timestamped log file.

Usage:
    from logger import setup_logger
    log = setup_logger("compute_means")   # writes to results/logs/compute_means_YYYYMMDD_HHMMSS.log
    log("something happened")
"""
import os
import sys
import datetime
from config import RESULTS_DIR


def setup_logger(name: str):
    """
    Returns a log function that writes to both stdout and a log file.

    Parameters
    ----------
    name : str
        Script name, used in the log filename.

    Returns
    -------
    log : callable
        Call log("message") anywhere instead of print("message").
    """
    log_dir = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    log_file = open(log_path, "w", buffering=1)  # line-buffered

    # Write header
    header = f"[{name}] started at {timestamp}\n{'='*60}\n"
    sys.stdout.write(header)
    sys.stdout.flush()
    log_file.write(header)

    def log(msg: str = ""):
        line = msg + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()

    log._path = log_path
    log._file = log_file
    print(f"[logger] Writing to {log_path}", flush=True)
    return log
