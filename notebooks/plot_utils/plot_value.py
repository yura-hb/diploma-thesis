import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .legend import add_legend


def plot_value(path: str | dict, info: dict, figsize=(8, 8), ax = None, post_process_fn=lambda a: a, background_process_fn=None):
    if not isinstance(path, dict):
        path = dict(first=path)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    min_value_len = float('inf')

    for name, data_path in path.items():
        if not isinstance(data_path, list):
            data_path = [data_path]

        result = []

        for p in data_path:
            df = pd.read_csv(p)

            if info.get('norm_index', False):
                df[info['index']] -= df[info['index']].min()

            if f := info.get('filter'):
                df = f(df)

            suffix = '' if len(path) == 1 else f'{name}'

            df = df.sort_values(by=info['index'])
            df.set_index(info['index'], inplace=True)

            result += [post_process_fn(df[info['column']])]

            min_value_len = min(min_value_len, len(result[-1]))


        if len(result) == 1:
            ax.plot(result[0], label=name)
        else:
            result = [v[:min_value_len] for v in result]
            result = np.vstack(result)

            min_value = result.min(axis=0)
            mean_value = result.mean(axis=0)
            max_value = result.max(axis=0)

            ax.plot(np.arange(len(mean_value)), mean_value, marker=info['marker'], label=name)

            ax.fill_between(np.arange(len(mean_value)), min_value, max_value, alpha=0.25)

    ax.grid(True, zorder=0)

    if 'title' in info:
        ax.set_title(info['title'])

    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])

    # if len(path) > 1:
    #     add_legend(ax, info)

    if ax is None:
        return fig
