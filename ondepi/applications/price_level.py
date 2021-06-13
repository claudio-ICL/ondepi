import pandas as pd
from ondepi.resources.queue.queue import Queue, produce_df_detection
from ondepi.resources.lobster import dataparser as dp
from ondepi.resources import utils
import numpy as np

logger = utils.get_logger()


def init_queue_on_lobster_data(
    symbol='INTC',
    date='2019-01-31',
    direction=1,
    price_level=466000,
    std_size=100,
    t0=0.0,
    t1=576000,
    reset_time_origin=False,
):
    info_str = "\n".join(
        [f"{k}: {v}" for k, v in locals().items() if isinstance(v, (str, int))])
    logger.info(info_str)
    queue = Queue()
    queue.set_price(price_level)
    sample = dp.parse_sample(symbol=symbol, date=date,
                             direction=direction, price_level=price_level, std_size=std_size,
                             t0=t0, t1=t1, reset_time_origin=reset_time_origin)
    queue.set_sample(sample)
    return queue


def calibrate_queue_on_lobster_data(
    queue=None,
    symbol='INTC',
    date='2019-01-31',
    direction=1,
    price_level=466000,
    std_size=100,
    t0=0.0,
    t1=576000,
    reset_time_origin=False,
):
    if queue is None:
        queue = init_queue_on_lobster_data(symbol=symbol, date=date,
                                           direction=direction, price_level=price_level, std_size=std_size,
                                           t0=t0, t1=t1, reset_time_origin=reset_time_origin)
    queue.calibrate_on_self(
        num_guesses=12,
        ftol=1e-14,
        gtol=1e-9,
        maxiter=2000,
        disp=0)
    return queue


def set_param(queue,
              param_D=(-1.59205528, -0.03106934,
                       9.30799028,  0.55593776,  0.03977957),
              param_A=(-1.1194079, -9.19896215,
                       7.58084436,  0.54751069064,  0.04019904)
              ):
    param_D = np.array(param_D, dtype=np.float64)
    param_A = np.array(param_A, dtype=np.float64)
    queue.set_param(param_D, param_A)
    return queue
