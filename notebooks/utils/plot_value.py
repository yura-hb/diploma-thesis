
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_value(path: str, info: dict, figsize=(8,8), post_process_fn = lambda a: a):
    if not os.path.exists(path):
        raise ValueError('No loss file exists')

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=figsize)

    if 'work_center_id' in df.columns:
        work_centers = df['work_center_id'].unique()
        machines = df['machine_id'].unique()

        for work_center_id in work_centers:
            for machine_id in machines:
                filtered = df[(df['work_center_id'] == work_center_id) & (df['machine_id'] == machine_id)]
                filtered = filtered.sort_values(by=info['index'])
                filtered.set_index(info['index'], inplace=True)

                if len(machines) == 1:
                    label = f'M_idx: {work_center_id}'
                else:
                    label = f'W_idx: {work_center_id}, M_idx: {machine_id}'

                ax.plot(post_process_fn(filtered[info['column']]), label=label)
    else:
        df.set_index(info['index'], inplace=True)

        ax.plot(post_process_fn(df[info['column']]))

    ax.grid(True)
    ax.set_title(info['title'])
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])

    ax.legend()

    return fig, df
