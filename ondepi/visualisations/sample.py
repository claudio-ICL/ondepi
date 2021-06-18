import matplotlib.pyplot as plt
import pandas as pd

"""
The following functions accept either 
a dataframe returned by 'ondepi.resources.lobster.dataparser.parse_df_sample',
or
numpy arrays representing times, events and states.
"""


def __init__(df, ax=None, time_range=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
    if isinstance(time_range, (tuple, list)):
        idx = (min(time_range) <= df['time']) & (
            df['time'] <= max(time_range))
        df = df.loc[idx, :].copy()
    return df, ax


def plot_reconstructed_volumes(df, ax=None, time_range=None, std_size=100, **kwargs):
    df, ax = __init__(df, ax, time_range)
    time = df['time'].values
    volumes = std_size * df['state'].values
    kwargs['width'] = kwargs.get('width', 7.5e-1)
    ax.bar(time, volumes, **kwargs)
    ax.set_title(
        f'Level volumes reconstructed from queue state')
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('volume')
    return ax


def plot_df_detection(df, ax=None, time_range=None, std_size=100, **kwargs):
    df['time'] = df['time sample']
    df, ax = __init__(df, ax, time_range)
    time = df['time'].values
    volumes = std_size * df['state'].values
    predictions = std_size * df['predictor'].values
    kwargs['width'] = kwargs.get('width', 7.5e-1)
    kwargs['label'] = kwargs.get('label', 'Queue size')
    ax.bar(time, volumes, **kwargs)
    ax.plot(time, predictions, color='green', linewidth=3.0,
            linestyle='--', label='Expected value')
    ax.set_title(
        f'Level volumes reconstructed from queue state and their expected values')
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('volume')
    ax.legend()
    return ax
