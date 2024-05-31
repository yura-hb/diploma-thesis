import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from .legend import add_legend

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def plot_value_per_run(path: str | dict, info: dict, make_run_path, post_process_fn=lambda a: a, ax = None):
    if not isinstance(path, dict):
        path = dict(first=path)

    fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=info.get('figsize', (8, 8)))

    for name, data_path in path.items():
        if not isinstance(data_path, list):
            data_path = [data_path]

        result = []

        min_value_len = float('inf')

        for p in data_path:
            try:
                run = 1

                values = []

                while True:
                    run_path = make_run_path(p, run)

                    run += 1

                    try:
                        df = pd.read_csv(run_path)
                        df = df.sort_values(by=info['index'])
                        df.set_index(info['index'], inplace=True)

                        values += [post_process_fn(df[info['column']], run)]
                    except:
                        break

                values = np.array(values)

                if value := info.get('smoothing_value'):
                    values = np.convolve(values, np.ones(value), 'valid') / value

                result += [values]

                min_value_len = min(min_value_len, len(values))
            except:
                pass

        if len(result) == 1:
            ax.plot(np.arange(len(result[0])), result[0], marker=info['marker'], label=name)
        else:
            result = [v[:min_value_len] for v in result]
            result = np.vstack(result)

            min_value = result.min(axis=0)
            mean_value = result.mean(axis=0)
            max_value = result.max(axis=0)

            ax.plot(np.arange(len(mean_value)), mean_value, marker=info['marker'], label=name)

            ax.fill_between(np.arange(len(mean_value)), min_value, max_value, alpha=0.25)

    ax.grid(True, zorder=0)
    ax.set_title(info['title'])
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    # ax.set_xticks(np.arange(max_values_len))

    # add_legend(ax, info)

    return fig
