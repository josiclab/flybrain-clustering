"""
utility functions for flybrain clustering
"""
import datetime
import sys
import re

# from scipy import sparse


def log_msg(*args, out=sys.stdout, **kwargs):
    """Print message m with a timestamp if out is not None."""
    if out:
        print(datetime.datetime.now().strftime("%Y %m %d %H:%M:%S "), *args, **kwargs, file=out)


def flatten(t):
    return [item for sublist in t for item in sublist]


def simplify_type(s):
    """
    Parses a cell type from the hemibrain database and extracts the first few
    letters or returns 'part' if the name starts with '('
    """
    if s[0] == '(':
        return 'part'
    elif s == "None":
        return s
    else:
        m = re.search(r'([A-Z]+)(.*)', s)
        return m.group(1)
