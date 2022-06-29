import sys
import logging
import os

FORCE_LOGGER_STDOUT = False
if 'FORCE_LOGGER_STDOUT' in os.environ:
    if os.environ['FORCE_LOGGER_STDOUT'] == '1':
        FORCE_LOGGER_STDOUT = True

def get_logger(name):

    logger = logging.getLogger(name)

    #print("get_logger() invoked with name:", name, "has handlers:", logger.handlers)

    logger.setLevel(logging.INFO)
    # https://stackoverflow.com/questions/533048/how-to-log-source-file-name-and-line-number-in-python
    formatter = logging.Formatter('%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    if FORCE_LOGGER_STDOUT:
        ch = logging.StreamHandler(sys.stdout)
    else:
        ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging
    logger.propagate = False

    logging.basicConfig(datefmt='%Y-%m-%d:%H:%M:%S')

    #logger.debug('debug message')
    #logger.info('info message')
    #logger.warn('warn message')
    #logger.error('error message')
    #logger.critical('critical message')

    return logger