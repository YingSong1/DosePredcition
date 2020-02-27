import os, logging, traceback
from collections import OrderedDict
import numpy as np

def get_logger(log_path):
    parent_path = os.path.dirname(log_path)  # get parent path
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(console)
    return logger