import logging
import os

DEFAULT_LEVEL = logging.DEBUG

LOG_FORMAT = '%(name)s %(asctime)s %(message)s'


class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + LOG_FORMAT + reset,
        logging.INFO: grey + LOG_FORMAT + reset,
        logging.WARNING: yellow + LOG_FORMAT + reset,
        logging.ERROR: red + LOG_FORMAT + reset,
        logging.CRITICAL: bold_red + LOG_FORMAT + reset
    }

    def __init__(self, date_format):
        super().__init__()
        self.date_format = date_format

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_format)
        return formatter.format(record)

from config import log_path

class LogWrapper:
    def __init__(self, name, mode="a"):
        self.create_directory()
        self.name = name
        self.filename = log_path + self.name + '.log'
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(DEFAULT_LEVEL)

        file_handler = logging.FileHandler(self.filename, mode=mode)
        stream_handler = logging.StreamHandler()
        formatter_file = logging.Formatter(LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter_file)
        stream_handler.setFormatter(ColorFormatter('%H:%M:%S'))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info("LogWrapper init() " + self.filename)

    @staticmethod
    def create_directory():
        path = '../logs'
        if not os.path.exists(path):
            os.makedirs(path)
