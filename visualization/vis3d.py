import numpy as np
import ipyvolume as ipv
import collections
import pandas as pd


def parse_mesh_bytes(b, encoding='utf-8'):
    """Given bytes (e.g. as returned by fetch_roi_mesh), parse into vertices and
    triangles. Returns (X,Y,Z,T) as in get_roi_mesh_from_file"""
    lines = b.decode(encoding).split('\n')
    vertices = get_vertices(lines)
    triangles = get_triangles(lines)
    return vertices[0], vertices[1], vertices[2], triangles


def get_vertices(thing):
    return np.array([[float(c) for c in ell.split()[1:]] for ell in thing if ell.startswith('v')]).T


def get_triangles(thing):
    return [[int(c) - 1 for c in ell.split()[1:]] for ell in thing if ell.startswith('f')]


def format_skeleton(s, node_col, parent_col, mode="skeleton",
                    x_col="x", y_col="y", z_col="z", radius_col="radius",
                    suffixes=["0", "1"]):
    """Given a dataframe `s` which contains skeleton information, format it for
    easy processing with ipyvolume's plot_trisurf.

    This means: First, map the node_col to the range [0,n], where n is the number of nodes
    Next, use the same mapping on parent_col.
    These will be put in s["node"] and s["parent"], respectively.

    Then, if mode='skeleton' (default), just return the formatted dataframe.
    If mode='mesh', merge the dataframe with itself along `node_col`, `parent_col`

    Example usage:
    s = format_skeleton(skel, 'rowId', 'link', mode='skeleton')
    ipv.plot_trisurf(s.x, s.y, s.z, lines=s[s.parent != -1].values)

    s = format_skeleton(skel, 'rowId', 'link', mode='mesh')
    for _, r in s.iterrows():
        V, T = cylinder([r.x0, r.y0, r.z0], [r.x1, r.y1, r.z1], r.radius0, r.radius1)
        ipv.plot_trisurf(V[0], V[1], V[2], triangles=T)"""
    s = pd.DataFrame(s)  # basically, copy it to simplify my life
    ids = list(np.unique(np.union1d(s[s[node_col] != -1][node_col], s[s[parent_col] != -1][parent_col])))
    s["node"] = s[node_col].apply(lambda i: index(ids, i))
    s["parent"] = s[parent_col].apply(lambda i: index(ids, i))

    if mode == "mesh":
        s = s.merge(s, left_on="parent", right_on="node", suffixes=suffixes, how="left")

    return s


def frenet_frame(P, Q):
    """Return a frenet frame with ihat pointing from P to Q"""
    ihat = Q - P
    ihat = ihat / np.linalg.norm(ihat)
    jhat = np.array([ihat[1], -ihat[0], 0])
    if (jhat ** 2).sum() == 0:  # PQ happens to point in the z direction
        jhat = np.array([1, 0, 0])  # any old vector in the xy plane does the trick
    khat = np.cross(ihat, jhat)
    return ihat, jhat, khat


def cylinder(P, Q, r0, r1=None, *, n=17):
    """Return vertices and triangles for a cylinder from `P` to `Q`
    With base radius r0. If r1 is None, the bases at P and Q have the same radius
    Otherwise, the base at P has radius r0 and the base at Q has radius r1.

    The base is a circle made of `n` points"""
    if r1 is None:
        r1 = r0
    P, Q = np.array(P), np.array(Q)
    ihat, jhat, khat = frenet_frame(P, Q)

    theta = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    VP = P[:, None] + r0 * np.cos(theta) * jhat[:, None] + r0 * np.sin(theta) * khat[:, None]
    VQ = Q[:, None] + r1 * np.cos(theta) * jhat[:, None] + r1 * np.sin(theta) * khat[:, None]
    V = np.hstack([VP, VQ, P[:, None], Q[:, None]])

    TPq = np.vstack([np.arange(n), np.roll(np.arange(n), -1), np.arange(n, 2 * n)])
    TQp = np.vstack([np.arange(n, 2 * n), np.roll(np.arange(n, 2 * n), 1), np.arange(n)])
    TP = np.vstack([np.ones(n, dtype=int) * 2 * n, np.arange(n), np.roll(np.arange(n), -1)])
    TQ = np.vstack([np.ones(n, dtype=int) * 2 * n + 1, np.arange(n, 2 * n), np.roll(np.arange(n, 2 * n), 1)])

    T = np.hstack([TPq, TQp, TP, TQ])

    return V, T.T


def cone(P, Q, r, *, n=17):
    """Return vertices and triangles for a cone with base centered at `P` and apex at `Q`

    The base is a circle of `n` points."""
    P, Q = np.array(P, dtype=float), np.array(Q, dtype=float)
    ihat, jhat, khat = frenet_frame(P, Q)

    theta = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    VP = P[:, None] + r * np.cos(theta) * jhat[:, None] + r * np.sin(theta) * khat[:, None]
    V = np.hstack([VP, P[:, None], Q[:, None]])

    Tbase = np.vstack([np.arange(n), np.roll(np.arange(n), -1), np.array([n] * n)])
    Tside = np.vstack([np.arange(n), np.roll(np.arange(n), -1), np.array([n + 1] * n)])
    T = np.hstack([Tbase, Tside])

    return V, T.T


def plot_axis_key(location, length, radius, n=34,
                  axis_shift=0.0,
                  u=[1, 0, 0], v=[0, 1, 0], w=[0, 0, 1],
                  colors=["red", "yellow", "blue"],
                  arrows=True, arrow_radius=None, arrow_length=None,
                  head="arrow", tail="cylinder"):
    """Use ipv to plot a 3d 'axis key', i.e. a little tripod
    of orthogonal axes aligned with the xyz axes.
    Change the axes/orientation by changing `u`, `v`, `w`

    axis_shift=k will translate the axes by k * length in the direction of u,v,w.
    For example, axis_shift=0 will give a "corner" style plot, while
    axis_shift=-0.5 will result in a more traditional six-pointed axis."""
    location = np.array(location, dtype=float)
    u = np.array(u) / np.linalg.norm(u)
    v = np.array(v) / np.linalg.norm(v)
    w = np.array(w) / np.linalg.norm(w)
    if not isinstance(length, (collections.abc.Sequence, np.ndarray)):
        length = [length] * 3
    if not isinstance(axis_shift, (collections.abc.Sequence, np.ndarray)):
        axis_shift = [axis_shift] * 3
    if arrow_radius is None:  # default arrow radius
        arrow_radius = 1.5 * radius
    if arrow_length is None:  # convert to list if needed
        arrow_length = [0.3 * ell for ell in length]
    if not isinstance(arrow_length, (collections.abc.Sequence, np.ndarray)):
        arrow_length = [arrow_length] * 3
    arrow_length = np.array(arrow_length, dtype=float)
    for vec, le, t, al, c in zip(map(np.array, [u, v, w]), length, axis_shift, arrow_length, colors):
        P, Q = location + t * le * vec, location + (1 + t) * le * vec
        V, T = cylinder(P, Q, radius, n=n)
        ipv.plot_trisurf(V[0], V[1], V[2], triangles=T, color=c)
        if head is not None:
            Ph, Qh = Q, Q + al * vec
            if head == "arrow":
                Vh, Th = cone(Ph, Qh, arrow_radius, n=n)
            else:  # default is a cylinder
                Vh, Th = cylinder(Ph, Qh, radius, n=n)
            ipv.plot_trisurf(Vh[0], Vh[1], Vh[2], triangles=Th, color=c)
        if tail is not None:
            Pt, Qt = P, P - al * vec
            if tail == "arrow":
                Vt, Tt = cone(Pt, Qt, arrow_radius, n=n)
            else:  # default is cylinder
                Vt, Tt = cylinder(Pt, Qt, radius, n=n)
            ipv.plot_trisurf(Vt[0], Vt[1], Vt[2], triangles=Tt, color=c)


def rotate(x, y, z, C="center", theta=np.pi / 2, plane="xy"):
    """Given a point cloud, rotate it around C. By default, compute C as the mean.
    By default, rotate 90 degrees in the xy plane."""
    X = np.vstack([x, y, z])
#     C = X.mean(axis=1)
    if isinstance(C, str):
        C = X.mean(axis=1)
    rot = np.eye(3)
    c0, c1 = "xyz".index(plane[0]), "xyz".index(plane[1])
    rot[c0, c0] = np.cos(theta)
    rot[c1, c0] = np.sin(theta)
    rot[c0, c1] = -np.sin(theta)
    rot[c1, c1] = np.cos(theta)
    tX = X - C[:, None]  # "let's write a python package for doing matrix algebra but make it a pain in the ass to do broadcast operations" -- numpy authors, probably
    tY = (rot @ (tX))
    Y = tY + C[:, None]
    return Y[0], Y[1], Y[2]


def index(items, obj, default=-1):
    """Get the index of `obj` in `items`, or return `default` if it's not in there.
    Uesful for making a custom sorting order, that forces unexpected items to on specific place."""
    try:
        return items.index(obj)
    except ValueError:
        return default
