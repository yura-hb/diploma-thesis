
from functools import reduce


class Queue:

    def __init__(self, is_distributed: bool):
        self.is_distributed = is_distributed
        self.queue = dict()

    def store(self, shop_floor_id, key, moment, record):
        self.queue[shop_floor_id] = self.queue.get(shop_floor_id, dict())

        if self.is_distributed:
            self.queue[shop_floor_id][key] = self.queue[shop_floor_id].get(key, dict())
            self.queue[shop_floor_id][key][moment] = record
        else:
            self.queue[shop_floor_id][moment] = self.queue[shop_floor_id].get(moment, []) + [record]

    def pop(self, shop_floor_id):
        if shop_floor_id not in self.queue:
            return None

        values = self.queue[shop_floor_id]

        del self.queue[shop_floor_id]

        return values

    def pop_group(self, shop_floor_id, key):
        if shop_floor_id not in self.queue:
            return None

        values = self.__get_group__(shop_floor_id, key)

        self.__del_group__(shop_floor_id, key)

        values = sorted(values.items(), key=lambda item: item[0])
        values = reduce(lambda acc, item: acc + item[1], values, [])

        return values

    def store_group(self, shop_floor_id, key, records):
        implicit_moment = -1

        self.queue[shop_floor_id] = self.queue.get(shop_floor_id, dict())

        if self.is_distributed:
            self.queue[shop_floor_id][key] = self.queue[shop_floor_id].get(key, dict())
            self.queue[shop_floor_id][key][implicit_moment] = records
        else:
            self.queue[shop_floor_id][implicit_moment] = records

    def group_len(self, shop_floor_id, key):
        group = self.__get_group__(shop_floor_id, key)

        if len(group) == 0:
            return 0

        return sum([len(records) for _, records in group.items()])

    def __get_group__(self, shop_floor_id, key):
        values = dict()

        if shop_floor_id not in self.queue:
            return values

        if self.is_distributed:
            if key not in self.queue[shop_floor_id]:
                return values

            return self.queue[shop_floor_id][key]

        return self.queue[shop_floor_id]

    def __del_group__(self, shop_floor_id, key):
        if shop_floor_id not in self.queue:
            return

        if self.is_distributed:
            if key not in self.queue[shop_floor_id]:
                return

            del self.queue[shop_floor_id][key]
            return

        del self.queue[shop_floor_id]

