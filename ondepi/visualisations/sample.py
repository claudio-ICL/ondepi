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


def plot_detection(df, ax=None, time_range=None, std_size=100, **kwargs):
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


def plot_impact(df, ax=None, time_range=None, direction=1, std_size=100, **kwargs):
    df['time'] = df['time sample']
    df, ax = __init__(df, ax, time_range)
    time = df['time'].values
    impact = direction * std_size * df['detector'].values
    kwargs['color'] = kwargs.get('color', 'lightpink')
    kwargs['label'] = kwargs.get('label', 'Queue impact')
    kwargs['alpha'] = kwargs.get('alpha', 0.7)
    ax.fill_between(time, impact, **kwargs)
    kwargs['alpha'] = max(1.0, 1.2 * kwargs.get('alpha', 1.0))
    ax.plot(time, impact, **kwargs)
    ax.set_title(
        f'Impact on level volumes')
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('volume')
    ax.legend()
    return ax


def plot_detection_and_impact(df, time_range=None, direction=1, std_size=100, width=1.e-1, **kwargs):
    fig = plt.figure(figsize=(10, 12))
    ax_detection = fig.add_subplot(211)
    ax_detection = plot_detection(
        df, ax=ax_detection, time_range=time_range, std_size=std_size, width=width)
    ax_impact = fig.add_subplot(212)
    ax_impact = plot_impact(
        df, ax=ax_impact, time_range=time_range, std_size=std_size, **kwargs)
    plt.show()
