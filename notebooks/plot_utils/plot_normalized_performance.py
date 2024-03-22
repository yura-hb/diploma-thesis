import matplotlib.pyplot as plt
import numpy as np


def plot_normalized_performance(df, info):
    index_column = info['index']
    metric = info['metric']
    candidate_column = info['candidate_column']
    baseline = info['baseline']

    fig, ax = plt.subplots(figsize=info.get('figsize', (12, 6)))

    candidates = info.get('candidates', np.sort(df[candidate_column].unique()))
    candidates = [candidate for candidate in candidates if candidate != baseline]

    print(candidates)

    baseline_info = df[df[candidate_column] == baseline].set_index(index_column)

    for index, candidate in enumerate(candidates):
        candidate_info = df[df[candidate_column] == candidate].set_index(index_column)

        delta = (baseline_info[metric] - candidate_info[metric]) / (baseline_info[metric] + 1e-10)
        delta = delta[~np.isnan(delta)]

        delta *= 100

        ax.boxplot(delta, positions=[index], notch=True, vert=True)

    ax.yaxis.grid(True)

    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_title(info['title'])

    ax.set_xticks(np.arange(len(candidates)), candidates, rotation=45)

    plt.tight_layout()

    return fig
