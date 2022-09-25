import numpy as np


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs(100*(x-y)/x))


def get_rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))