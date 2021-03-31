"""
Utility functions for visualization routines in vis.py
"""
import numpy as np

from math import sqrt, ceil


def circle_arc(P, Q, R, k):
    """Given a circle centered at `P` with points `Q` and `R` on the circle,
    returns the arc of the circle from Q to R. Produces `2 ** k` points.

    `P`, `Q`, `R` should be numpy arrays with 2 elements.

    Assumes that the user has checked that indeed `Q` and `R` lie on the same circle centered at `P`."""
    PQ, PR = Q - P, R - P
    costheta = PQ.dot(PR) / np.linalg.norm(PQ) / np.linalg.norm(PR)

    cosdtheta = costheta
    for i in range(k - 1):
        cosdtheta = sqrt((1 + cosdtheta) / 2)

    sindtheta = sqrt((1 - cosdtheta) / 2)
    cosdtheta = sqrt((1 + cosdtheta) / 2)

    sgn = np.sign(PQ[0] * PR[1] - PQ[1] * PR[0])
    rot = np.array([[cosdtheta, - sgn * sindtheta], [sgn * sindtheta, cosdtheta]])
    xs, ys = np.zeros(1 + (2 ** k)), np.zeros(1 + (2 ** k))
    xs[0], ys[0] = Q[0], Q[1]
    v = np.array(PQ)
    for i in range(1, 1 + (2**k)):
        v = rot.dot(v)
        xs[i], ys[i] = v[0] + P[0], v[1] + P[1]

    return xs, ys


def inverted_circle_arc(P, Q, R, k, diag_tol=1e-8):
    """Given a circle centered at `P` and points `Q`, `R` on the circle, draw the arc of the "inverse circle"
    through `Q` and `R`, i.e. the circle that intersects at right angles. If the points are close enough to
    diammetrically opposed, return a straight line segment."""
    if ((Q + R - 2 * P) ** 2).sum() < diag_tol:
        return np.array([Q[0], R[0]]), np.array([Q[1], R[1]])  # points are close enough to antipodal; return straight line

    # compute the new center
    PQ = Q - P
    M = (Q + R) / 2
    PM = M - P
    Pp = P + PM * (PQ ** 2).sum() / (PM ** 2).sum()

    # return circle arc centered at new center
    return circle_arc(Pp, Q, R, k)


def flowchart_quarter_circle_curve(P, Q, b, circle_k=3):
    """
    Returns the xs and ys for a flowchart curve that moves from P to Q, drawn
    using straight line segments joined by quarter-circle arcs. `b` is the
    length of the horizontal segment.
    """
    half_height = np.abs(Q[1] - P[1]) / 2
    r = min(half_height, np.abs(Q[0] - P[0]) / 2 - b)
    if r <= 0:
        return np.array([P[0], Q[0]]), np.array([[P[1], Q[1]]])
    # exclue_vertical = (r != half_height)

    bvec = np.array([np.sign(Q[0] - P[0]) * b, 0])
    r_displacement = np.sign(Q[1] - P[1]) * r

    # xs, ys = [P[0]], [P[1]]
    Pp = P + bvec
    C1 = P + bvec + np.array([0, r_displacement])
    Ppp = np.array([(P[0] + Q[0]) / 2, P[1] + r_displacement])
    Qpp = np.array([(P[0] + Q[0]) / 2, Q[1] - r_displacement])
    C2 = Q - bvec - np.array([0, r_displacement])
    Qp = Q - bvec

    first_circle_xs, first_circle_ys = circle_arc(C1, Pp, Ppp, circle_k)
    second_circle_xs, second_circle_ys = circle_arc(C2, Qpp, Qp, circle_k)
    # if r == half_height:
    #     second_circle_xs, second_circle_ys = second_circle_xs[1:], second_circle_ys[1:]
    xs = np.hstack([[P[0]], first_circle_xs, second_circle_xs, Q[0]])
    ys = np.hstack([[P[1]], first_circle_ys, second_circle_ys, Q[1]])
    return xs, ys


def index_to_unique_list(index, name):
    """
    Convert the given pandas index into a list of strings by concatenating with the given name.
    """
    return [str(name) + str(idx) for idx in index]


def repeat_to_match_lengths(list_to_repeat, length_to_match):
    """Return `list_to_repeat * n`, truncated to length `length_to_match`"""
    list_length = len(list_to_repeat)
    n = int(ceil(length_to_match / list_length))
    return (list_to_repeat * n)[:length_to_match]


def tuple_to_string(t):
    """
    Given a multi-index tuple, shortens it for the heatmap plot.
    """
    if "pre" in t[0]:
        return str(t[1]) + "->"
    elif "post" in t[0]:
        return "->" + str(t[1])
    else:
        return str(t[1])
