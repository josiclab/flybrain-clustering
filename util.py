"""
utility functions for flybrain clustering
"""
import datetime
import sys
import re
import numpy as np

# from scipy import sparse


def log_msg(*args, out=sys.stdout, **kwargs):
    """Print message m with a timestamp if out is not None."""
    if out:
        print(datetime.datetime.now().strftime("%Y %m %d %H:%M:%S "), *args, **kwargs, file=out)


def flatten(t):
    return [item for sublist in t for item in sublist]


def unique(items):
    """Sometimes I want the unique elements in a list, but np.unique() doesn't work how I want."""
    return [x for i, x in enumerate(items) if x not in items[:i]]


def swap(d, u, v):
    """Given a dictionary d, swap the values in keys u and v
    (yes, this is literally just the line `d[u], d[v] = d[v], d[u]`)"""
    d[u], d[v] = d[v], d[u]


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
        m = re.search(r'([a-zA-Z_]+)(.*)', s)
        ans = m.group(1)
        if ans in ["v", "l"]:
            return s[:4]
        if ans == "s":
            return s
        return ans


def joint_marginal(df, c1, c2, include_fraction=False):
    """Given a dataframe and two columns, return a dataframe with the joint and marginal counts."""
    j = df.value_counts([c1, c2])
    j.name = "joint_count"
    j = j.reset_index()

    m1 = df.value_counts(c1)
    m1.name = f"{c1}_count"
    j = j.merge(m1, left_on=c1, right_index=True)

    m2 = df.value_counts(c2)
    m2.name = f"{c2}_count"
    j = j.merge(m2, left_on=c2, right_index=True)

    if include_fraction:
        j["joint_fraction"] = j["joint_count"] / j["joint_count"].sum()
        j[f"{c1}_fraction"] = j["joint_count"] / j[f"{c1}_count"]
        j[f"{c2}_fraction"] = j["joint_count"] / j[f"{c2}_count"]
    return j


def accumulate_clusters(fs, threshold=0.9, parts=0):
    """Given a data series, return the first indices such that their sum passes threshold

    If parts > 0, try to make a list with that many parts."""
    s = 0
    fs = fs.sort_values(ascending=False)
    idxs = []
    vs = []
    for idx, v in zip(fs.index, fs):
        s += v
        idxs.append(idx)
        vs.append(v)
        if s >= threshold:
            break
    # Now, repeat indices to get the right number of parts.
    if len(idxs) > 1 and parts > 0:
        # if we need multiple clusters to reach threshold
        # and we want proportional shading, we need to do some fiddling
        vs = np.array(vs)
        vs = vs / vs.sum()
        idxs_ = []
        for p, (idx, v) in enumerate(zip(idxs, vs)):
            n_reps = int(round(v * parts))
            if p == 0:
                n_reps = int(np.ceil(v * parts))
            idxs_ += [idx] * n_reps
        idxs = idxs_
    return idxs
