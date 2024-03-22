
import numpy as np
import matplotlib.pyplot as plt

from .legend import add_legend


def plot_performance_across_runs(data, info):
    metric = info['metric']
    group = info['group']
    candidate_column = info['candidate_column']
    candidates = info['candidates']

    fig, ax = plt.subplots(figsize=info.get('figsize', (12, 6)))

    runs = np.sort(data[group].unique())
    runs = list(runs)

    run_to_id = {run: i for i, run in enumerate(runs)}

    data['run_id'] = data[group].apply(lambda x: run_to_id[x])

    for run in np.sort(data[group].unique()):
        filtered = data[data[group] == run]

        ax.plot(filtered[metric], filtered['run_id'], marker=info['marker'], ls='', c='gray')

    if isinstance(candidates, list):
        for candidate in candidates:
            filtered = data[data[candidate_column] == candidate]

            ax.plot(filtered[metric],
                    filtered['run_id'],
                    marker=info['marker'],
                    ms=10,
                    ls='',
                    label=candidate)

    if isinstance(candidates, dict):
        for title, candidates in candidates.items():
            filtered = data[data[candidate_column].isin(candidates)]

            ax.plot(filtered[metric],
                    filtered['run_id'],
                    marker=info['marker'],
                    ms=10,
                    ls='',
                    label=title)


    ax.xaxis.grid(True)

    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_title(info['title'])

    ax.set_yticks(np.arange(len(runs)), runs)

    if len(candidates) > 1:
        add_legend(ax, info)

    plt.tight_layout()

    return fig
