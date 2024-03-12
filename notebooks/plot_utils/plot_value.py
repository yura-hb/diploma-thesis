import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .legend import add_legend


def plot_value(path: str | dict, info: dict, figsize=(8, 8), post_process_fn=lambda a: a):
    if not isinstance(path, dict):
        path = dict(first=path)

    fig, ax = plt.subplots(figsize=figsize)

    for name, data_path in path.items():
        df = pd.read_csv(data_path)

        if info.get('norm_index', False):
            df[info['index']] -= df[info['index']].min()

        suffix = '' if len(path) == 1 else f'{name}'

        if 'work_center_id' in df.columns and info.get('is_reward_per_unit_visible', False):
            work_centers = np.sort(df['work_center_id'].unique())
            machines = np.sort(df['machine_id'].unique())

            for work_center_id in work_centers:
                for machine_id in machines:
                    filtered = df[(df['work_center_id'] == work_center_id) & (df['machine_id'] == machine_id)]
                    filtered = filtered.sort_values(by=info['index'])
                    filtered.set_index(info['index'], inplace=True)

                    if len(machines) == 1:
                        label = f'M_idx: {work_center_id}'
                    else:
                        label = f'W_idx: {work_center_id}, M_idx: {machine_id}'

                    if len(suffix) > 0:
                        label += ' ' + suffix

                    ax.plot(post_process_fn(filtered[info['column']]), label=label)
        else:
            df = df.sort_values(by=info['index'])
            df.set_index(info['index'], inplace=True)

            ax.plot(post_process_fn(df[info['column']]), label=name)

    ax.grid(True)
    ax.set_title(info['title'])
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])

    add_legend(ax, info)

    return fig
