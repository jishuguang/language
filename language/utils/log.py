import logging

LOG_FORMAT = "[%(asctime)s][%(levelname)s]%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def get_logger():
    return logging.getLogger()
