"""
logger.py — Tees all output to both terminal and a timestamped log file.

Usage:
    from utils.logger import setup_logger
    log = setup_logger("compute_means", log_dir="./results/logs")
    log("something happened")
"""
import os
import sys
import datetime


def setup_logger(name: str, log_dir: str = None):
    """
    Returns a log function that writes to both stdout and a log file.

    Parameters
    ----------
    name : str
        Script name, used in the log filename.
    log_dir : str or None
        Directory for log files. If None, logs only to stdout.

    Returns
    -------
    log : callable
        Call log("message") anywhere instead of print("message").
    """
    log_file = None

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
        log_file = open(log_path, "w", buffering=1)  # line-buffered

        # Write header
        header = f"[{name}] started at {timestamp}\n{'='*60}\n"
        sys.stdout.write(header)
        sys.stdout.flush()
        log_file.write(header)

        print(f"[logger] Writing to {log_path}", flush=True)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        header = f"[{name}] started at {timestamp}\n{'='*60}\n"
        sys.stdout.write(header)
        sys.stdout.flush()

    def log(msg: str = ""):
        line = msg + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_file is not None:
            log_file.write(line)
            log_file.flush()

    if log_file is not None:
        log._path = log_path if log_dir else None
        log._file = log_file
    else:
        log._path = None
        log._file = None

    return log
