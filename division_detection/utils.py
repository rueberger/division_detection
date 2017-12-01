""" Contains utility methods
"""

import itertools
import math
import os
import numpy as np
import json

from warnings import warn


def round_arr_up(arr):
    """ Round array elements to nearest int, with behavior that .5 always rounds up

    Args:
       arr: array

    Returns:
       rounded_arr: with dtype int
    """
    arr_rounder = np.vectorize(round_up)
    return arr_rounder(arr).astype(int)

def round_up(num):
    """ Round num to nearest int, .5 rounds up
    """
    return int(num + math.copysign(0.5, num))

def take(n, iterable):
    """ Taken from itertools cookbook
    Return first n items of the iterable as a list
    """
    while True:
        yield list(itertools.islice(iterable, n))

def vol_to_int8(vol):
    """ Normalizes and encodes volume as uint8 for visualization
    """
    # normed and centered [0, 1]
    norm_vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    return (norm_vol * 255).astype(np.uint8)

def clean_dir(dir_name):
    """ Deletes the contents of dir_name
    """
    print("WARNING: cleaning {}".format(dir_name))
    for file_name in os.listdir(dir_name):
        os.remove('{}/{}'.format(dir_name, file_name))

def get_search_history(model_name):
    """ Parses and returns the search history

    Returns:
       losses - list
       n_hidden_us - list
    """
    log_dir = os.path.expanduser('~/results/div_detect/search/{}/loss'.format(model_name))
    losses = []
    n_hus = []
    for job_log in os.listdir(log_dir):
        try:
            with open('{}/{}'.format(log_dir, job_log), 'r') as log_file:
                log = json.load(log_file)
                losses.append(log['loss'])
                n_hus.append(log['n_conv_hu'])
        except Exception as err:
            warn("Caught while trying to parse {}: {}".format(job_log, err))
    return losses, n_hus

def setup_logging(log_name, log_path):
    """ Sets up module level logging
    """
    import logging
    # define module level logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.expanduser(log_path)

    # define file handler for module
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    # create formatter and add to handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(fh)

    # TODO: implement email handler
    # define email handler for important logs in module
    #eh = logging.SMTPHandler()
    return logger


def package_path():
    """ Returns the absolute path to this package base directory

    """
    package_name = 'division_detection'
    import sys
    mjhmc_path = [path for path in sys.path if package_name in path][0]
    if mjhmc_path is None:
        raise Exception('You must include {} in your PYTHON_PATH'.format(package_name))
    prefix = mjhmc_path.split(package_name)[0]
    return "{}{}".format(prefix, package_name)
