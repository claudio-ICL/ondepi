import matplotlib.pyplot as plt
import pandas as pd
from ondepi.resources.utils import check_nonempty_df

"""
The following functions accept a dataframe 
returned by 'ondepi.resources.lobster.dataparser.parse_price_level'.
"""


def __init__(df, ax=None, time_range=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
    if isinstance(time_range, (tuple, list)):
        idx = (min(time_range) <= df['time']) & (
            df['time'] <= max(time_range))
        df = df.loc[idx, :].copy()
    check_nonempty_df(df)
    return df, ax


def scatter_event_label(df, ax=None, time_range=None):
    df, ax = __init__(df, ax, time_range)
    time = df['time'].values
    event_labels = df['event_label'].values
    ax.scatter(time, event_labels, c='blue')
    ax.set_title('Time series of events as labelled in LOOBSTER')
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('event label')
    return ax


def plot_level_volume(df, ax=None, time_range=None, **kwargs):
    df, ax = __init__(df, ax, time_range)
    direction = 'Bid' if df['direction'].min() == 1 else 'Ask'
    price_level = int(df['price_level'].min())
    time = df['time'].values
    volumes = df['level_volume'].values
    kwargs['width'] = kwargs.get('width', 7.5e-1)
    ax.bar(time, volumes, **kwargs)
    ax.set_title(
        f'{direction} volumes at price level {price_level}')
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('volume')
    return ax


def plot_level_evolution(df, ax=None, time_range=None, **kwargs):
    df, ax = __init__(df, ax, time_range)
    direction = 'Bid' if df['direction'].min() == 1 else 'Ask'
    price_level = int(df['price_level'].min())
    time = df['time'].values
    level = df['level'].values
    kwargs['where'] = kwargs.get('where', 'post')
    ax.step(time, level, **kwargs)
    ax.set_title(
        f'{direction} level of price {price_level} (1 + distance from best {direction})')
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('level')
    return ax
