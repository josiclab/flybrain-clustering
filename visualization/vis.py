"""
Functions for visualizing network data using `bokeh` and `matplotlib` packages.
"""
import numpy as np
import pandas as pd
import colorcet as cc

import matplotlib.pyplot as plt

from bokeh.plotting import figure
from bokeh.models import (Rect, MultiLine, Circle,
                          GraphRenderer, StaticLayoutProvider,
                          NodesAndLinkedEdges,
                          HoverTool, TapTool, ColumnDataSource,
                          LinearColorMapper, LogColorMapper,
                          ColorBar, BasicTicker)
from bokeh.transform import transform, factor_cmap

from math import sin, pi, sqrt, ceil


def circle_layout_graph(node_df, edge_df,
                        node_data_cols=[],
                        node_index_name="id", use_node_df_index=True,
                        node_fill_by="index", node_line_by="index", node_line_width=3,
                        default_line_color="#111111",
                        hover_fill_color="#00ff00", hover_line_color="#00aa00",
                        selected_fill_color="#ff0000", selected_line_color="#ff0000",
                        edge_start_col="node1", edge_end_col="node2", edge_weight_col="total_weight",
                        edge_data_cols=[], use_alpha=True, log_weight=True,
                        node_fill_palette=cc.glasbey_dark, node_line_palette=cc.glasbey_light,
                        layout_origin=np.zeros(2), layout_radius=1.0,
                        circular_arcs=True, circle_k=3,
                        hover_tooltips={"id": "@index"}):
    """
    Return a `bokeh.GraphRenderer` object and list of tools that display the graph specified by the input dataframes.

    Required arguments are `node_df`, `edge_df` which specify the structure of the graph.
    The column `node_df[node_index_name]` or `node.index` will be used as the graph renderer index.
    Other columns can be stored in the graph renderer by specifying a list `node_data_cols` (default empty list `[]`).
    `node_fill_by` and `node_line_by` specify how colors are chosen for the nodes. Valid options are the default "index",
    or a list of columns in `node_df`. In the first case, the specified palettes[1] will be repeated to match the length
    of the node index. Otherwise each unique combination of values in those columns will be mapped to a color in the
    palette (repeating if necessary). The dataframe will be sorted by the fill color columns, if specified.

    [1] A `bokeh` palette is a list of strings, each a hex code for a color; see https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    """
    graph = GraphRenderer()

    n_nodes = node_df.shape[0]
    # set up fill color, if specified
    if node_fill_by == "index":
        node_df["fill_color"] = repeat_to_match_lengths(node_fill_palette, n_nodes)
    elif node_fill_by is not None:
        node_df = node_df.sort_values(by=node_fill_by)
        uniques_df = pd.DataFrame(node_df[node_fill_by].value_counts())
        uniques_df["fill_color"] = repeat_to_match_lengths(node_fill_palette, uniques_df.shape[0])
        node_df = node_df.merge(uniques_df[["fill_color"]], left_on=node_fill_by, right_index=True)
        del uniques_df
    if "fill_color" in node_df.columns and "fill_color" not in node_data_cols:
        node_data_cols.append("fill_color")

    # set up line color, if specified
    if node_line_by == "index":
        node_df["line_color"] = repeat_to_match_lengths(node_line_palette, n_nodes)
    elif node_line_by is not None:
        uniques_df = pd.DataFrame(node_df[node_line_by].value_counts())
        uniques_df["line_color"] = repeat_to_match_lengths(node_line_palette, uniques_df.shape[0])
        node_df = node_df.merge(uniques_df[["line_color"]], left_on=node_line_by, right_index=True)
        del uniques_df
    if "line_color" in node_df.columns and "line_color" not in node_data_cols:
        node_data_cols.append("line_color")

    # Use the node DF as the data source for the node renderer
    if len(node_data_cols) == 0:
        node_data_cols = node_df.columns
    graph.node_renderer.data_source.data = node_df[node_data_cols]
    if use_node_df_index:
        node_index = node_df.index
    else:
        node_index = node_df[node_index_name]
    graph.node_renderer.data_source.data["index"] = node_index

    # add node layout info
    if "theta" not in node_df.columns:
        theta = np.linspace(0, 2 * np.pi, n_nodes + 1)[:-1]
    else:
        theta = node_df['theta']
    nodes_x = layout_origin[0] + layout_radius * np.cos(theta)
    nodes_y = layout_origin[1] + layout_radius * np.sin(theta)
    graph_layout = dict(zip(node_index, zip(nodes_x, nodes_y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # add edges
    edge_data = {"start": edge_df[edge_start_col], "end": edge_df[edge_end_col], **{k: edge_df[k] for k in edge_data_cols}}
    if edge_weight_col is not None:
        edge_data["weight"] = edge_df[edge_weight_col]
    else:
        edge_data["weight"] = np.ones(edge_df.shape[0])
    if log_weight:
        edge_data["weight"] = np.log(edge_data["weight"]) + 1
    if use_alpha:
        edge_data["alpha"] = edge_data["weight"] / np.max(edge_data["weight"])
    graph.edge_renderer.data_source.data = edge_data

    # style the nodes
    graph.node_renderer.glyph = Circle(radius=sin(2 * pi / n_nodes / 3), fill_color="fill_color", line_color="line_color", line_width=node_line_width)
    graph.node_renderer.hover_glyph = Circle(radius=sin(2 * pi / n_nodes / 3), fill_color=hover_fill_color, line_color=hover_line_color, line_width=node_line_width)
    graph.node_renderer.selection_glyph = Circle(radius=sin(2 * pi / n_nodes / 3), fill_color=selected_fill_color, line_color=selected_line_color, line_width=node_line_width)

    graph.edge_renderer.glyph = MultiLine(line_color=default_line_color, line_width="weight", line_alpha="alpha")
    graph.edge_renderer.hover_glyph = MultiLine(line_color=hover_line_color, line_width="weight", line_alpha=1.0)
    graph.edge_renderer.selection_glyph = MultiLine(line_color=selected_line_color, line_width="weight", line_alpha=1.0)

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    if circular_arcs:
        xs, ys = [], []
        for start_index, end_index in zip(graph.edge_renderer.data_source.data["start"], graph.edge_renderer.data_source.data["end"]):
            Q = np.array(graph_layout[start_index])
            R = np.array(graph_layout[end_index])
            circle_xs, circle_ys = inverted_circle_arc(layout_origin, Q, R, circle_k)
            xs.append(circle_xs)
            ys.append(circle_ys)
        graph.edge_renderer.data_source.data['xs'] = xs
        graph.edge_renderer.data_source.data['ys'] = ys

    tools = [TapTool(), HoverTool(tooltips=[(k, v) for k, v in hover_tooltips.items()])]

    return graph, tools


def breakdown_flowchart_graph(df, columns=None, x_coords=None,
                              bar_width=1, gap=3,
                              palette=cc.glasbey_dark,
                              hover_line_color="#ff0000", line_cap="butt",
                              line_width_mode="clamp",
                              max_line_width=100, circle_k=3,
                              hover_tooltips={}):
    """
    Given a dataframe with categorical data across columns, produce a breakdown
    figure which shows how things regroup from one column to the next. By
    default, uses all columns in `data_df` which can potentially cause problems

    `line_width_mode = "clamp"` will apply min(w, max_line_width) to the edge
    widths, any other option will just use the actual edge width (might make
    the graph unreadable though)
    """
    graph = GraphRenderer()

    # handle missing parameters
    if columns is None:
        columns = df.columns
    if x_coords is None:
        x_coords = [gap * i for i in range(len(columns))]

    # set up the node data
    node_index = []
    node_x, node_y, node_height = [], [], []
    col_name, col_value = [], []
    for c, x in zip(columns, x_coords):
        val_counts = df[c].value_counts()
        new_indices = index_to_unique_list(val_counts.index, c)
        node_index += new_indices
        col_name += [c] * len(new_indices)
        col_value += val_counts.index.to_list()
        node_x += [x] * len(new_indices)
        node_y += list(val_counts.values.cumsum() - val_counts.values / 2)
        node_height += val_counts.values.tolist()
    n_nodes = len(node_index)

    graph = GraphRenderer()
    palette = repeat_to_match_lengths(palette, n_nodes)

    # add the node renderer
    graph.node_renderer.data_source.data = {"index": node_index,
                                            "col_name": col_name,
                                            "col_value": col_value,
                                            "height": node_height,
                                            "color": palette,
                                            "node_x": node_x,
                                            "node_y": node_y}
    graph_layout = dict(zip(node_index, zip(node_x, node_y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    color_mapper = dict(zip(node_index, palette))  # used for coloring edges

    # style the nodes
    graph.node_renderer.glyph = Rect(width=bar_width, height="height", fill_color="color")
    graph.node_renderer.hover_glyph = Rect(width=bar_width, height="height", fill_color="color", line_color=hover_line_color, line_width=2)
    graph.node_renderer.selection_glyph = Rect(width=bar_width, height="height", fill_color="color")

    # construct the edges and their paths
    start, end = [], []
    xs, ys = [], []
    edge_width, edge_color = [], []
    for c0, c1 in zip(columns[:-1], columns[1:]):
        vc = df[[c0, c1]].value_counts()
        new_starts = index_to_unique_list(vc.index.get_level_values(0), c0)
        new_ends = index_to_unique_list(vc.index.get_level_values(1), c1)
        for s, e in zip(new_starts, new_ends):
            P = np.array(graph_layout[s])
            Q = np.array(graph_layout[e])
            curve_xs, curve_ys = flowchart_quarter_circle_curve(P, Q, bar_width / 2, circle_k)
            xs.append(curve_xs)
            ys.append(curve_ys)
        start += new_starts
        end += new_ends
        if line_width_mode == "clamp":
            edge_width += [min(v, max_line_width) for v in vc.values]
        else:
            edge_width += vc.values.tolist()
        edge_color += [color_mapper[s] for s in new_starts]

    # add the edge data to the renderer
    graph.edge_renderer.data_source.data = {"start": start, "end": end,
                                            "line_width": edge_width,
                                            "xs": xs, "ys": ys,
                                            "color": edge_color}
    graph.edge_renderer.glyph = MultiLine(line_width="line_width", line_color="color", line_alpha=0.5, line_cap=line_cap)
    graph.edge_renderer.hover_glyph = MultiLine(line_width="line_width", line_color="color", line_alpha=1.0, line_cap=line_cap)
    graph.edge_renderer.selection_glyph = MultiLine(line_width="line_width", line_color="color", line_alpha=1.0, line_cap=line_cap)
    graph.edge_renderer.nonselection_glyph = MultiLine(line_width="line_width", line_color="color", line_alpha=0.2, line_cap=line_cap)

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    tools = [TapTool(), HoverTool(tooltips=[(k, v) for k, v in hover_tooltips.items()])]

    suggested_x_range = (min(node_x) - bar_width / 2, max(node_x) + bar_width / 2)
    suggested_y_range = (0, max(y + h / 2 for y, h in zip(node_y, node_height)))

    return graph, tools, (suggested_x_range, suggested_y_range)


def breakdown_barchart_figure(node_df, select_col, select_value,
                              columns=[],
                              legend_threshold=0, x_coords=None,
                              figsize=(20, 20)):
    """
    Produce the "fraction-represented" bar chart for the given nodes.
    `select_col` and `select_value` are used to select the nodes of interest:
    The selected df is `node_df[node_df[select_col] == select_value]`
    `columns` is which columns are actually represented in the figure.
    """
    if len(columns) == 0:
        return None

    if x_coords is None:
        x_coords = np.arange(len(columns))

    celltypes = []
    instances = []

    f = plt.figure(figsize=figsize)

    restrict_df = node_df[columns]
    select_df = node_df[node_df[select_col] == select_value][columns]
    for x, c in zip(x_coords, columns):
        vc = select_df[c].value_counts()
        full_counts = restrict_df[restrict_df[c].isin(vc.index)][c].value_counts()
        for cluster, count, cumulative in zip(vc.index, vc.values, vc.values.cumsum()):
            fraction = max(count / full_counts[cluster], 0.01)
            leg_ent = str(cluster) + " (" + str(count) + ")"

            plt.bar([x], [count], bottom=cumulative - count, width=0.8, linewidth=1, edgecolor="black", color="white", label="_nolegend_")
            p = plt.bar([x], [count], bottom=cumulative - count, width=0.8 * fraction, label=leg_ent)
            if c == "celltype" and count >= legend_threshold:
                celltypes.append(p)
            if c == "instance" and count >= legend_threshold:
                instances.append(p)

    l1 = plt.legend(title="Cell Type", bbox_to_anchor=(1.01, 1), loc='upper left', handles=celltypes, fontsize=12)
    plt.legend(title="Instances", bbox_to_anchor=(1.15, 1), loc="upper left", handles=instances, fontsize=12)
    plt.gca().add_artist(l1)

    plt.xticks(x_coords, columns, fontsize=12)
    plt.xlabel("Parameter")
    plt.ylabel("# Cells in cluster")
    plt.title("Parameter " + str(select_col) + " Cluster " + str(select_value) + " grouping by algorithm parameters", fontsize=20)

    return f


def code_heatmap(full_df, codes, node_header="node", node_data=[],
                 continuous_palette=cc.fire, category_palette=cc.glasbey,
                 color_mapping="linear",
                 fig_title="Cluster codes",
                 width=800, height=1000,
                 node_font_size="7px", category_font_size="17px"):
    """
    Produce a plot like the one below. The first columns are specified by `node_data`, the latter are specified by `codes`.
    While `codes` and `node_data` will be used to filter out columns from `code_df`, all rows will be used, so be sure to
    select only the rows you want before passing it to this function.
    """

    type_df = full_df[node_header][node_data]
    type_df.index = type_df.index.astype(str)
    x_categories = list(type_df.columns)
    if len(node_data) > 0:
        y_categories = type_df.sort_values(by=node_data).index.to_list()
    else:
        y_categories = list(full_df.index.astype(str))
    type_stack = pd.DataFrame(type_df.stack(), columns=["type"]).reset_index()
    type_source = ColumnDataSource(type_stack)

    code_df = full_df[codes]
    code_df.index = code_df.index.astype(str)
    code_df.columns = pd.Series([tuple_to_string(c) for c in code_df.columns], name="neighbor")
    x_categories += list(code_df.columns)
    code_df = pd.DataFrame(code_df.stack(), columns=["code"]).reset_index()
    code_source = ColumnDataSource(code_df)

    if color_mapping == "log":
        code_df["code"] = code_df["code"] + 1
        mapper = LogColorMapper(palette=continuous_palette, low=code_df["code"].min(), high=code_df["code"].max())
    else:
        mapper = LinearColorMapper(palette=continuous_palette, low=code_df["code"].min(), high=code_df["code"].max())

    p = figure(title=fig_title,
               plot_width=width, plot_height=height,
               x_range=x_categories, y_range=y_categories,
               x_axis_location="above")
    p.rect(x="neighbor", y="id",
           width=1, height=1,
           source=code_source,
           line_color=None,
           fill_color=transform("code", mapper))
    category_palette = repeat_to_match_lengths(category_palette, len(type_stack["type"].unique()))
    p.rect(x="parameter", y="id", width=1, height=1, source=type_source, line_color=None, fill_color=factor_cmap("type", palette=category_palette, factors=list(type_stack["type"].unique())))

    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=10))
    p.add_layout(color_bar, "right")

    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.yaxis.major_label_text_font_size = node_font_size
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    p.xaxis.major_label_text_font_size = category_font_size
    # p.grid.grid_line_width = 0.0

    return p


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
