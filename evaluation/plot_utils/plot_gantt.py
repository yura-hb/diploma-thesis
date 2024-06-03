
import pandas as pd
import matplotlib.pyplot as plt


def plot_gantt(df: pd.DataFrame, info: dict, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    start_moment, end_moment = info['window']
    production_logs = df

    for block in info['blocks']:
        start_event, end_event = block['events']

        windows = production_logs[(production_logs['moment'] < end_moment) & (production_logs['moment'] >= start_moment)].copy()
        windows = windows[(windows['event'] == start_event) | (windows['event'] == end_event)]

        for event in [start_event, end_event]:
            windows.loc[windows['event'] == event, 'index'] = range(len(windows[windows['event'] == event]))

        windows = windows.set_index(['index', 'job_id', 'operation_id', 'work_center_idx', 'machine_idx'])

        result = windows[windows['event'] == start_event].copy()
        result['end_moment'] = windows[windows['event'] == end_event]['moment']

        result.drop(['event'], axis=1, inplace=True)
        result = result.reset_index()

        result['end_moment'] = result['end_moment'].fillna(end_moment)

        for job_id in result['job_id'].unique():
            job = result[result['job_id'] == job_id]

            ax.barh(
                y=job['work_center_idx'],
                width=job['end_moment'] - job['moment'],
                left=job['moment'],
                label=f'Job {job_id}',
                **block['style']
            )

    ax.set_title(info['title'])
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])

    return fig
