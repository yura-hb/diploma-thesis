
import logging


class Loggable:

    def __init__(self):
        self.logger = None

    def with_logger(self, logger: logging.Logger):
        self.logger = logger.getChild(self.__class__.__name__)

        return self

    def log(self, message, level: str = 'info'):
        if hasattr(self, 'logger'):
            self.logger.log(level, message)
        else:
            print(message)

    def log_debug(self, message):
        self.log(message, logging.DEBUG)

    def log_info(self, message):
        self.log(message, logging.INFO)

    def log_warning(self, message):
        self.log(message, logging.WARNING)

    def log_error(self, message):
        self.log(message, logging.ERROR)
