import matplotlib.pyplot as plt
import pandas as pd

"""
The following functions accept either 
a dataframe returned by 'ondepi.resources.lobster.dataparser.parse_df_sample',
or
numpy arrays representing times, events and states.
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


def plot_reconstructed_volumes(df, ax=None, time_i_range=None, std_size=100, **kwargs):
    df, ax = __init__(df, ax, time_i_range)
    time_i = df['time_i'].values
    volumes = std_size * df['state'].values
    kwargs['width'] = kwargs.get('width', 2.5e9)
    ax.bar(time_i, volumes, **kwargs)
    ax.set_title(
        f'Level volumes reconstructed from queue state')
    ax.set_xlabel('time in microseconds')
    ax.set_ylabel('volume')
    return ax
