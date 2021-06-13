import numpy as np
import pandas as pd
from functools import lru_cache
from ondepi.settings import data_path
from ondepi.resources.utils import (
        get_logger,
        check_nonempty_df,
        timestamp_to_idx,
        idx_to_timestamp,
        arrays_to_sample, 
        sample_to_arrays,
)

logger = get_logger()

def lob_header(int n_levels=10, first_col_time=False):
    cdef list cols = [
            'ask_price_1',
            'ask_volume_1',
            'bid_price_1',
            'bid_volume_1'
        ]
    if (first_col_time):
        cols.insert(0, 'time')
    for i in range(2, n_levels + 1):
        cols.append('ask_price_{}'.format(i))
        cols.append('ask_volume_{}'.format(i))
        cols.append('bid_price_{}'.format(i))
        cols.append('bid_volume_{}'.format(i))
    return cols

def mf_header():
    cols = [
        'time',
        'event_label',
        'order_id',
        'size',
        'price',
        'direction'
    ]
    return cols

def merge_lob_and_mf(
        lob, mf, **kwargs):
    kwargs['left_index'] = kwargs.get('left_index', True)
    kwargs['right_index'] = kwargs.get('right_index', True)
    kwargs['validate'] = kwargs.get('validate', '1:1')
    df = mf.merge(lob, **kwargs)
    return df

@lru_cache()
def load_mf(str symbol='INTC', str date='2019-01-31'):
    cdef str file_name = f'{symbol}_{date}_34200000_57600000_message_10.csv'
    file_path = data_path / f'{symbol}' / file_name
    if not file_path.exists():
        logger.error(f"File path {str(file_path)} does not exist")
        raise FileNotFoundError()
    with open(file_path, 'rb') as source:
        df = pd.read_csv(file_path, header=None, index_col=False, usecols = list(range(6)))
    df.columns = mf_header()
    astype_dict = {col: 'int' for col in df.columns if 'time' not in col.lower()}
    df = df.astype(astype_dict)
    df.insert(0, 'time_i', timestamp_to_idx(np.array(df['time'].values, dtype=np.float64)))
    return df
    
@lru_cache()
def load_lob(str symbol='INTC', str date='2019-01-31'):
    cdef str file_name = f'{symbol}_{date}_34200000_57600000_orderbook_10.csv'
    file_path = data_path / f'{symbol}' / file_name
    if not file_path.exists():
        logger.error(f"File path {str(file_path)} does not exist")
        raise FileNotFoundError()
    with open(file_path, 'rb') as source:
        df = pd.read_csv(file_path, header=None, index_col=False, names=lob_header()) 
    return df

def load_and_merge(str symbol='INTC', str date='2019-01-31', **kwargs):
    return merge_lob_and_mf(
            load_lob(symbol=symbol, date=date),
            load_mf(symbol=symbol, date=date),
            **kwargs)

def declare_level(df, long ticksize = 100):
    df.insert(0, 'bid_price_ante', df['bid_price_1'].shift(-1).ffill(downcast='infer').values)
    df.insert(0, 'ask_price_ante', df['ask_price_1'].shift(-1).ffill(downcast='infer').values)
    df.insert(6, 'level', np.zeros(len(df), dtype=np.int64))
    idx_ask = (df['direction'] == -1)
    idx_bid = (df['direction'] == 1)
    df.loc[idx_ask, 'level'] =\
        1 + np.array(
        (df.loc[idx_ask, 'price'] -
         df.loc[idx_ask, 'ask_price_ante']) // ticksize,
        dtype=np.int64)
    df.loc[idx_bid, 'level'] =\
        1 + np.array(
        (-df.loc[idx_bid, 'price'] +
         df.loc[idx_bid, 'bid_price_ante']) // ticksize,
        dtype=np.int64)
    df.drop(columns=['bid_price_ante', 'ask_price_ante'], inplace=True)
    return df

@lru_cache()
def read_full_dataframe(
        str symbol='INTC',
        str date='2019-01-31',
        **kwargs
):
    df = load_and_merge(symbol=symbol, date=date, **kwargs)
    df = declare_level(df)
    # Set time origin
    df['time'] -= df['time'].min()
    df['time_i'] -= df['time_i'].min()
    return df

def select_price_level(df, long price_level = 468000):
    cdef list mf_cols = [col for col in df.columns if col not in lob_header()]
    cdef list price_cols = [col for col in lob_header() if 'price' in col.lower()]
    cdef list volume_cols = [col for col in lob_header() if 'volume' in col.lower()]
    cdef np.ndarray[long, ndim=2] arr = np.array(df.loc[:, price_cols].values, dtype=np.int64)
    idx = arr==price_level
    cdef np.ndarray[long, ndim=1] level_volume = np.array(
            df.loc[:, volume_cols].values[idx], dtype=np.int64)
    idx_ = np.array(np.sum(idx, axis=1), dtype=bool)
    df_ = df.loc[idx_, mf_cols].copy()
    df_.insert(df_.shape[1], 'price_level', price_level)
    df_.insert(df_.shape[1], 'level_volume', level_volume)
    return df_

def select_time_window(
        df, 
        double t0=0.0,
        double t1=57600.0,
        reset_time_origin=False,
):
    idx = (t0 <= df['time']) & (df['time'] <= t1)
    df = df.loc[idx, :].copy()
    if reset_time_origin:
        df['time'] -= t0
        df['time_i'] -= <long>floor(t0 * 10**9)
    return df

def load_queue(
        df = None,
        long direction = 1,
        long price_level = 468000,
        str symbol='INTC',
        str date='2019-01-31',
        double t0=0.0,
        double t1=57600.0,
        reset_time_origin=False,
        **kwargs
):
    if df is None:
        df = read_full_dataframe(symbol=symbol, date=date, **kwargs)
    idx = (df['direction']==direction) & (df['price']==price_level)
    df = select_price_level(df.loc[idx, :].copy(), price_level=price_level)
    df = select_time_window(df, t0, t1, reset_time_origin)
    return df

def is_time_increasing(df):
    cdef np.ndarray[double, ndim=1] time = np.array(df['time'].values, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] dt = np.diff(time)
    return np.all(dt>0.0)

def drop_same_time_i(df):
    return df.groupby(['time_i']).last().reset_index()

def derive_df_event(df, EventType type_, standard_size=None):
    if standard_size is None:
        standard_size = int(np.quantile(df['size'], 0.50))
    cdef long std_size = <long>standard_size
    name = 'N_A' if type_==EventType.A else 'N_D'    
    labels = [1] if type_==EventType.A else [2, 3, 4]
    df = df.loc[df['event_label'].isin(labels), :].copy()
    df.sort_values(by='time_i', inplace=True)
    df.insert(0, name, np.cumsum(df['size'].values, dtype=np.int64) // std_size)
    agg_dict = {
            'time_i': 'last',
            'time': 'last',
            'level': 'min',
            'size': 'sum',
            'price': 'first',
            'direction': 'first',
            'price_level': 'max',
            'level_volume': 'last'
            }
    df = df.groupby(name).agg(agg_dict).reset_index()
    df.insert(3, 'event_type', type_)
    return df

cdef vector[long] define_event_times(
        vector[long] time_i,
        vector[long] dN,
        ):
    cdef long unsigned int size = time_i.size()
    assert dN.size() == size
    cdef vector[long] times_i
    times_i.reserve(2 * size)
    cdef long time, k
    cdef long unsigned int t
    for t in range(size):
        time = time_i.at(t)
        for k in range(dN.at(t)):
            times_i.push_back(time + k)
    return times_i

def define_events(
        df,
        EventType type_,
        long std_size=100,
        ):
    df = derive_df_event(df, type_, standard_size=std_size)
    cdef vector[long] time_i = df['time_i'].tolist()
    name = 'N_A' if type_==EventType.A else 'N_D'    
    cdef vector[long] dN = np.diff(df[name].values, prepend=0).tolist()
    cdef vector[long] times_i = define_event_times(time_i, dN)
    df = pd.DataFrame({
        'time_i': times_i,
        name: np.arange(times_i.size(), dtype=np.int64),
        'event': type_
        })
    return df

def join_events(df_D, df_A):
    df = pd.concat([df_D, df_A], axis=0, sort=True)
    df.sort_values(by='time_i', inplace=True)
    df = df.ffill(downcast='infer').bfill(downcast='infer')
    df['N_A'] = df['N_A'].astype(np.int64)
    df['N_D'] = df['N_D'].astype(np.int64)
    return df

def remove_negative_states(df):
    if 'state' not in df.columns:
        error_str = "'state' not found in columns {}".format(df.columns.tolist())
        logger.error(error_str)
        raise KeyError('state')
    to_drop = df['state'] < 0
    idx_to_drop = df.loc[to_drop, :].index
    df = df.drop(index=idx_to_drop).reset_index()
    return df

