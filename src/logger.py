import logging
import os


def get_logger(name=__name__):
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")

    logger = logging.getLogger(f'{name}_logger')
    logger.setLevel(logging.DEBUG)

    # Handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'./logs/logfile.log')
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return(logger)
