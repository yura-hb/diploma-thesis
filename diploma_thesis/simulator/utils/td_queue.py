
import heapq

from functools import reduce
from .queue import Queue


class TDQueue:

    def __init__(self, is_distributed: bool):
        self.is_distributed = is_distributed
        self.queue = dict()

    def reserve(self, shop_floor_id, key, moment):
        self.queue[shop_floor_id] = self.queue.get(shop_floor_id, dict())
        self.queue[shop_floor_id][key] = self.queue[shop_floor_id].get(key, dict())
        self.queue[shop_floor_id][key][moment] = None

    def store(self, shop_floor_id, key, moment, record):
        self.queue[shop_floor_id] = self.queue.get(shop_floor_id, dict())
        self.queue[shop_floor_id][key] = self.queue[shop_floor_id].get(key, dict())
        self.queue[shop_floor_id][key][moment] = record

    def store_slice(self, shop_floor_id, key, records):
        self.queue[shop_floor_id] = self.queue.get(shop_floor_id, dict())
        self.queue[shop_floor_id][key] = self.queue[shop_floor_id].get(key, dict())
        self.queue[shop_floor_id][key].update(records)

        d = self.queue[shop_floor_id][key]

        self.queue[shop_floor_id][key] = {k: d[k] for k in sorted(d.keys())}

    def prefix(self, shop_floor_id, key, length):
        if self.is_distributed:
            result = self.__prefix_for_key__(shop_floor_id, key, length)
        else:
            result = self.__prefix_for_shop_floor__(shop_floor_id, length)

        if result is None:
            return None

        return result

    def __prefix_for_key__(self, shop_floor_id, key, length):
        storage = self.queue.get(shop_floor_id, dict()).get(key, dict())

        if len(storage) < length:
            return None

        result = dict()

        while len(result) < length:
            moment, record = next(iter(storage.items()))

            if record is None:
                self.__revert__(shop_floor_id, {key: result})
                return None

            result[moment] = record

            del storage[moment]

        return sorted(result.items(), key=lambda item: item[0])

    def __prefix_for_shop_floor__(self, shop_floor_id, length):
        storage = self.queue.get(shop_floor_id, dict())

        if len(storage) == 0:
            return None

        records = 0
        result = dict()

        while records < length:
            min_moments = {k: min(v.keys()) for k, v in storage.items() if len(v) > 0}

            if len(min_moments) == 0:
                self.__revert__(shop_floor_id, result)
                return None

            key, moment = min(min_moments.items(), key=lambda item: item[1])

            if len(storage[key]) == 0:
                self.__revert__(shop_floor_id, result)
                return None

            record = storage[key][moment]

            if record is None:
                self.__revert__(shop_floor_id, result)
                return None

            del storage[key][moment]

            result[key] = result.get(key, dict())
            result[key][moment] = record

            records += 1

        result = reduce(lambda acc, item: acc + list(item[1].items()), result.items(), [])

        return sorted(result, key=lambda item: item[0])

    def __revert__(self, shop_floor_id, storage):
        for key, values in storage.items():
            self.store_slice(shop_floor_id, key, values)
