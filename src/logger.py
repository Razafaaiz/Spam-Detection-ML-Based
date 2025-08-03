import logging
import os
from datetime import datetime

# Create a directory for log files with a timestamped filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filemode='a'  # Append mode
)

# Optional: configure logging to also show logs in the console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
