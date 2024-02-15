

def filter(fn):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if fn(self, *args, **kwargs):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
