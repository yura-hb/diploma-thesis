
import cloudpickle


def save(path: str, obj: object):
    with open(path, 'wb') as file:
        cloudpickle.dump(obj, file)


def load(path: str) -> object:
    with open(path, 'rb') as file:
        return cloudpickle.load(file)
