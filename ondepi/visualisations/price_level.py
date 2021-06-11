import matplotlib.pyplot as plt
import pandas as pd

"""
The following functions accept a dataframe 
returned by 'ondepi.resources.lobster.dataparser.parse_price_level'.
"""


def __init__(df, ax=None, time_i_range=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
    if isinstance(time_i_range, (tuple, list)):
        idx = (min(time_i_range) <= df['time_i']) & (
            df['time_i'] <= max(time_i_range))
        df = df.loc[idx, :].copy()
    return df, ax


def scatter_event_label(df, ax=None, time_i_range=None):
    df, ax = __init__(df, ax, time_i_range)
    time_i = df['time_i'].values
    event_labels = df['event_label'].values
    ax.scatter(time_i, event_labels, c='blue')
    ax.set_title('Time series of events as labelled in LOOBSTER')
    ax.set_xlabel('time in microseconds')
    ax.set_ylabel('event label')
    return ax


def plot_level_volume(df, ax=None, time_i_range=None, **kwargs):
    df, ax = __init__(df, ax, time_i_range)
    direction = 'Bid' if df['direction'].min() == 1 else 'Ask'
    price_level = int(df['price_level'].min())
    time_i = df['time_i'].values
    volumes = df['level_volume'].values
    kwargs['width'] = kwargs.get('width', 2.5e9)
    ax.bar(time_i, volumes, **kwargs)
    ax.set_title(
        f'{direction} volumes at price level {price_level}')
    ax.set_xlabel('time in microseconds')
    ax.set_ylabel('volume')
    return ax


def plot_level_evolution(df, ax=None, time_i_range=None, **kwargs):
    df, ax = __init__(df, ax, time_i_range)
    direction = 'Bid' if df['direction'].min() == 1 else 'Ask'
    price_level = int(df['price_level'].min())
    time_i = df['time_i'].values
    level = df['level'].values
    kwargs['where'] = kwargs.get('where', 'post')
    ax.step(time_i, level, **kwargs)
    ax.set_title(
        f'{direction} level of price {price_level} (1 + distance from best {direction})')
    ax.set_xlabel('time in microseconds')
    ax.set_ylabel('level')
    return ax
