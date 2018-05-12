import time

# https://stackoverflow.com/a/3620972
PROF_DATA = {}


class Profile(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, fn):
        def with_profiling(*args, **kwargs):
            global PROF_DATA
            start_time = time.time()
            ret = fn(*args, **kwargs)

            elapsed_time = time.time() - start_time
            key = '[' + self.prefix + '].' + fn.__name__

            if key not in PROF_DATA:
                PROF_DATA[key] = [0, list()]
            PROF_DATA[key][0] += 1
            PROF_DATA[key][1].append(elapsed_time)

            return ret

        return with_profiling


def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
