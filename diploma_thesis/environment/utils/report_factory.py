from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd
from tabulate import tabulate

import environment


@dataclass
class Report:
    machines: pd.DataFrame
    shopfloor: pd.DataFrame

    def __str__(self):
        machines_table = self.__machine_table__()
        shop_floor_table = self.__shop_floor_table__()

        return str(shop_floor_table) + '\n' + str(machines_table)

    def __machine_table__(self):
        table = []

        for work_center_idx in self.machines['work_center_idx'].unique():
            data = self.machines[self.machines['work_center_idx'] == work_center_idx]

            grouped = data.groupby(['work_center_idx'])
            methods = ['min', 'mean', 'max']

            for idx, key in enumerate(methods):
                values = grouped.aggregate(key).reset_index()

                index = ""

                if idx == 0:
                    index += f'Work Center: { work_center_idx } '

                index += f'Agg: {key:>4}'
                values['index'] = index
                values.drop(['work_center_idx', 'machine_idx'], inplace=True, axis=1)

                table += [values.iloc[0].to_dict()]

            for machine_idx in data['machine_idx'].unique():
                machine_data = data[data['machine_idx'] == machine_idx].copy()
                machine_data['index'] = f'Machine: { machine_idx }'
                machine_data.drop(['work_center_idx', 'machine_idx'], inplace=True, axis=1)

                table += [machine_data.iloc[0].to_dict()]

        table = pd.DataFrame(table).set_index('index')

        return self.__make_table__(table)

    def __shop_floor_table__(self):
        return self.__make_table__(self.shopfloor)

    def __make_table__(self, table):
        return tabulate(table, headers='keys', tablefmt='psql', colalign=("right",))


class ReportFactory:

    def __init__(self,
                 statistics: 'environment.Statistics',
                 shop_floor: 'environment.ShopFloor',
                 time_predicate: 'environment.Statistics.Predicate.TimePredicate'):
        self.statistics = statistics
        self.shop_floor = shop_floor
        self.time_predicate = time_predicate

    def make(self) -> Report:
        machine_records = self.__make_machine_records__()
        shop_floor_records = self.__make_shopfloor_stats__()

        return Report(machine_records, shop_floor_records)

    def __make_machine_records__(self):
        records = []

        for machine in self.shop_floor.machines:
            worker_predicate = environment.Statistics.Predicate.MachinePredicate(
                machine_idx=machine.state.machine_idx, work_center_idx=machine.state.work_center_idx
            )

            records += [
                OrderedDict(
                    work_center_idx=machine.state.work_center_idx,
                    machine_idx=machine.state.machine_idx,
                    runtime=self.statistics.run_time(time_predicate=self.time_predicate, predicate=worker_predicate),
                    utilization_rate=self.statistics.utilization_rate(
                        time_predicate=self.time_predicate, predicate=worker_predicate
                    ),
                    number_of_processed_operations=self.statistics.total_number_of_processed_operations(
                        time_predicate=self.time_predicate, predicate=worker_predicate
                    )
                )
            ]

        return pd.DataFrame(records)

    def __make_shopfloor_stats__(self):
        predicate = environment.Statistics.Predicate(time_predicate=self.time_predicate)

        shop_floor = []

        for weighted_by_priority in [True, False]:
            shop_floor += [dict(
                weighted_by_priority=weighted_by_priority,
                total_jobs=len(self.statistics.jobs(predicate=predicate)),
                makespan=self.statistics.total_make_span(predicate=predicate),
                flow_time=self.statistics.total_flow_time(
                    weighted_by_priority=weighted_by_priority, predicate=predicate
                ),
                tardiness=self.statistics.total_tardiness(
                    weighted_by_priority=weighted_by_priority, predicate=predicate
                ),
                tardy_jobs=self.statistics.total_number_of_tardy_jobs(
                    weighted_by_priority=weighted_by_priority, predicate=predicate
                ),
                earliness=self.statistics.total_earliness(
                    weighted_by_priority=weighted_by_priority, predicate=predicate
                ),
            )]

        shop_floor = pd.DataFrame(shop_floor)

        return shop_floor
