import gc
import time
import traceback


def task(name_fn):
    def f(func):
        def wrapper(*args, **kwargs):
            name = name_fn(*args, **kwargs)
            start = time.time()

            print(f'Task started {name}')

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print(f'Error in task {name}: {e}')
                print(traceback.format_exc())
                return None

            print(f'Task finished { name }  Elapsed time: {time.time() - start} seconds.')

            gc.collect()

            return result

        return wrapper

    return f
