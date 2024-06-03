
import matplotlib.pyplot as plt
import numpy as np

from .legend import add_legend


def plot_reward_per_model_across_runs(data, info):
    metric = info['metric']
    reward = info['reward']
    group = info['group']
    candidate_column = info['candidate_column']
    candidates = info['candidates']

    fig, ax = plt.subplots(figsize=info.get('figsize', (12, 6)))

    for run in np.sort(data[group].unique()):
        filtered = data[data[group] == run]

        ax.plot(filtered[metric], filtered[reward], marker=info['marker'], ls='', c='gray')

    if isinstance(candidates, list) or isinstance(candidates, set):
        for candidate in candidates:
            filtered = data[data[candidate_column] == candidate]

            ax.plot(filtered[metric], filtered[reward], marker=info['marker'], ms=10, ls='', label=candidate)

    if isinstance(candidates, dict):
        for title, candidates in candidates.items():
            filtered = data[data[candidate_column].isin(candidates)]

            ax.plot(filtered[metric], filtered[reward], marker=info['marker'], ms=10, ls='', label=title)

    ax.grid(True, zorder=0)

    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_title(info['title'])

    if len(candidates) > 1:
        add_legend(ax, info)

    plt.tight_layout()

    return fig