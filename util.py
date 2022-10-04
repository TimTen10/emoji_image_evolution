import functools
import time


def timed_func(func):
    """
    Print the runtime of the decorated function.
    For a more precise measurement use the timeit module.
    """
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_func
