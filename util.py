"""
utility functions for flybrain clustering
"""
import datetime
import sys


# from scipy import sparse

def log_msg(*args, out=sys.stdout, **kwargs):
    """Print message m with a timestamp if out is not None."""
    if out:
        print(datetime.datetime.now().strftime("%Y %m %d %H:%M:%S "), *args, **kwargs, file=out)


def flatten(t):
    return [item for sublist in t for item in sublist]
