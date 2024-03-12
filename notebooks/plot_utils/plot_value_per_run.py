import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .legend import add_legend


def plot_value_per_run(path: str | dict, info: dict, make_run_path, post_process_fn=lambda a: a):
    if not isinstance(path, dict):
        path = dict(first=path)

    fig, ax = plt.subplots(figsize=info.get('figsize', (8, 8)))

    max_values_len = 0

    for name, data_path in path.items():
        run = 1

        values = []

        while True:
            run_path = make_run_path(data_path, run)

            run += 1

            try:
                df = pd.read_csv(run_path)

                df = df.sort_values(by=info['index'])
                df.set_index(info['index'], inplace=True)

                values += [post_process_fn(df[info['column']], run)]
            except:
                break

        ax.plot(np.arange(len(values)), np.array(values), marker=info['marker'], label=name)

        max_values_len = max(max_values_len, len(values))

    ax.grid(True)
    ax.set_title(info['title'])
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_xticks(np.arange(max_values_len))

    add_legend(ax, info)

    return fig
