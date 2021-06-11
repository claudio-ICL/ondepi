import numpy as np
import pandas as pd
from ondepi.resources.lobster import utils


def parse_sample(
        long direction = 1,
        long price_level = 468000,
        long std_size = 100,
        str symbol='INTC',
        str date='2019-01-31',
        **kwargs
):
    df = parse_df_sample(
             direction=direction,
             price_level=price_level,
             std_size=std_size,
             symbol=symbol,
             date=date,
             **kwargs
         )
    cdef vector[double] times = df['time'].tolist()
    cdef vector[EventType] events = df['event'].tolist()
    cdef vector[long] states = df['state'].tolist()
    cdef Sample sample = utils.arrays_to_sample(times, events, states)
    return sample 


def parse_df_sample(
        long direction = 1,
        long price_level = 468000,
        long std_size = 100,
        str symbol='INTC',
        str date='2019-01-31',
        **kwargs
):
    df = parse_price_level(direction=direction, price_level=price_level, 
            symbol=symbol, date=date, **kwargs)
    df.reset_index(inplace=True, drop=True)
    df_D = utils.define_events(df, EventType.D, std_size)
    df_A = utils.define_events(df, EventType.A, std_size)
    df_s = utils.join_events(df_D, df_A)
    cdef np.ndarray[double, ndim=1] times = utils.convert_idx_to_timestamp(
            np.array(df_s['time_i'].values, dtype=np.int64))
    df_s.insert(0, 'time', times)
    df_s.insert(1, 'state', np.array(df_s['N_A'].values - df_s['N_D'].values, dtype=np.int64))
    df_s = df_s[['time_i', 'time', 'event', 'state', 'N_D', 'N_A']]
    return df_s


def parse_price_level(
        df=None,
        long direction=1,
        long price_level=468000,
        str symbol='INTC',
        str date='2019-01-31',
        **kwargs
):
    df = utils.load_queue(df=df, direction=direction, price_level=price_level, 
            symbol=symbol, date=date, **kwargs)

    # Keep only events of interest
    to_drop = df['event_label'].isin([5, 6, 7])
    idx_to_drop = df.loc[to_drop, :].index
    df.drop(index=idx_to_drop, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Drop 'order_id'
    df.drop(columns=['order_id'], inplace=True, errors='ignore')

    # aggregate
    groupby_cols = ['time_i', 'event_label']
    agg_dict = {
            'time': 'last',
            'level': 'min',
            'size': 'sum',
            'price': 'min' if direction == 1 else 'max',
            'direction': 'first',
            'price_level': 'max',
            'level_volume': 'last'
            }
    df = df.groupby(groupby_cols).agg(agg_dict).reset_index()
    df = utils.drop_same_time_i(df)
    return df
