import logging
import pathlib
from datetime import datetime

# create a logger obj
infologger = logging.getLogger(__name__)
infologger.setLevel(logging.INFO)
# customize the format of log message
# here s means value will be formated as string
formatter = logging.Formatter(
    "%(asctime)s [ %(levelname)s ] - %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
)
# formating a log filename
file_name = f'{datetime.now().strftime("%d%b%y-%H.%M.%S")}.log'

# validating path for logs
log_dir_path = pathlib.Path(__file__).parent.parent.as_posix() + "/logs"
pathlib.Path(log_dir_path).mkdir(parents=True, exist_ok=True)
# formating the full file name
log_file_name = pathlib.Path(log_dir_path + f"/{file_name}")

# creting filehandler obj, will write log to files
file_handler = logging.FileHandler(log_file_name)
# set message format
file_handler.setFormatter(formatter)

# adding filehandler to logger
infologger.addHandler(file_handler)

if __name__ == "__main__":
    infologger.info("Testing logger")

# -------- ERROR --------
# ERROR: Can not execute `setup.py` since setuptools is not available in the build environment.
# Sol: just uninstall & install setuptools.
