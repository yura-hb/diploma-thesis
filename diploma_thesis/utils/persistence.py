
import pickle


def save(path: str, obj: object):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load(path: str) -> object:
    with open(path, 'rb') as file:
        return pickle.load(file)
