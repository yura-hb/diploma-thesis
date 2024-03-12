
import numpy as np
import matplotlib.pyplot as plt

from .legend import add_legend

def plot_reward_per_run(data, info, format_group):
    metric = info['metric']
    reward = info['reward']
    group = info['group']

    fig, ax = plt.subplots(figsize=info.get('figsize', (12, 6)))

    for run in np.sort(data[group].unique()):
        filtered = data[data[group] == run]

        ax.plot(filtered[metric], filtered[reward], marker=info['marker'], ls='', label=format_group(run))

    ax.grid(True)

    add_legend(ax, info)

    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_title(info['title'])

    plt.tight_layout()

    return fig