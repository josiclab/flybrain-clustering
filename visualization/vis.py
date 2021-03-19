"""
Utility functions for visualizing fly brain data using `bokeh` package.
"""
import numpy as np
import pandas as pd
import bokeh
import colorcet as cc
# import hvplot.pandas
# import holoviews as hv


# import bokeh.palettes
# from bokeh.plotting import figure
# from bokeh.io import output_file, show
from bokeh.models import (Ellipse, Rect, MultiLine, Circle,
                          GraphRenderer, StaticLayoutProvider,
                          EdgesAndLinkedNodes, NodesAndLinkedEdges,
                          HoverTool, BoxZoomTool, ResetTool, BoxSelectTool, TapTool,
                          ColumnDataSource)

from math import sqrt, cos, sin, pi, ceil


def circle_arc(P, Q, R, k):
    """Given a circle centered at `P` with points `Q` and `R` on the circle,
    returns the arc of the circle from P to Q. Produces `2 ** k` points.

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


def repeat_to_match_lengths(list_to_repeat, length_to_match,):
    """Return `list_to_repeat * n`, truncated to length `length_to_match`"""
    list_length = len(list_to_repeat)
    n = int(ceil(length_to_match / list_length))
    return (list_to_repeat * n)[:length_to_match]


def circle_layout_graph(node_df, edge_df,
                        node_data_cols=[],
                        node_index_name="id", use_node_df_index=True,
                        node_fill_by="index", node_line_by="index", node_line_width=3,
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

    graph.edge_renderer.glyph = MultiLine(line_color="#000000", line_width="weight", line_alpha="alpha")
    graph.edge_renderer.hover_glyph = MultiLine(line_color=hover_line_color, line_width="weight", line_alpha=1.0)
    graph.edge_renderer.selection_glyph = MultiLine(line_color=selected_line_color, line_alpha=1.0)

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


def circle_layout_graph_old(node_df, edge_df,
                            node_index_name="id", use_node_df_index=True,
                            fill_nodes_by=None, fill_color_function=bokeh.palettes.turbo,
                            line_color_nodes_by=None, line_color_palette=bokeh.palettes.Colorblind8,
                            node_data_cols={},
                            node_properties={"line_width": 3},
                            hover_node_properties={"line_width": 3},
                            selected_node_properties={"line_width": 3},
                            edge_start="node1", edge_end="node2", edge_data_cols={},
                            edge_weight="total_weight",  # edge_df column with edge weight
                            normalize_alpha=True,  # use varying alpha values for edges
                            edge_properties={},
                            hover_edge_properties={},
                            selected_edge_properties={},
                            layout_origin=np.zeros(2), layout_radius=1.0,
                            circular_arcs=True, circle_k=3,
                            add_hovertool=True, hover_tooltips={}):
    """Return a bokeh.GraphRenderer object and a list of tools that can plot the specified graph.

    `node_data_cols` should be a dictionary of `graph_layout_name: dataframe_column_name` pairs,
    for example: `{"index": "unique_neuron_id", "fillcolor": "cellcolor"}.
    Likewise, `edge_data_cols` specifies things the same way for edges.

    Returns `graph, tools` where `graph` is a renderer (e.g. append it to a `plot`'s list of renderers)
    and `tools` is a list of tools (e.g. the hover tool, zoom box tool, etc.)
    """
    graph = GraphRenderer()

    # add columns to dataframes as needed
    if "theta" not in node_df.columns:
        theta = np.linspace(0, 2 * np.pi, node_df.shape[0] + 1)[:-1]
        node_df['theta'] = theta

    # map colors to nodes -- this chunk is particular to the fly brain data
    if fill_nodes_by is not None:
        # color_nodes_by should be a list of columns of the node_df
        fill_color_mapper = pd.DataFrame(node_df[fill_nodes_by].value_counts())
        fill_color_mapper["fillcolor"] = fill_color_function(color_mapper.shape[0])

        node_df = node_df.reset_index().merge(fill_color_mapper, left_on=fill_nodes_by, right_index=True).set_index(node_index_name)
        node_properties["fill_color"] = "fillcolor"
    if line_color_nodes_by is not None:
        line_color_mapper = pd.DataFrame(node_df[line_color_nodes_by].unique())
        line_color_mapper["line_color"] = repeat_to_match_lengths(line_color_palette, line_color_mapper.shape[0])

        node_df = node_df.reset_index().merge(line_color_mapper, left_on=line_color_nodes_by, right_on=0).set_index(node_index_name)

    # add nodes from node_df
    graph.node_renderer.data_source.data = {k: node_df[v] for k, v in node_data_cols.items()}
#     grahph.node_renderer.data_source.data = bokeh.models.ColumnDataSource(node_df)
    if use_node_df_index:  # overwrites existing index, if defined!
        graph.node_renderer.data_source.data["index"] = node_df.index
    node_index = graph.node_renderer.data_source.data["index"]  # it'll be useful later to have this named
    n_nodes = len(node_index)  # also useful down the road

    # add node layout info
    nodes_x = layout_origin[0] + layout_radius * np.cos(node_df['theta'])
    nodes_y = layout_origin[1] + layout_radius * np.sin(node_df['theta'])
    graph_layout = dict(zip(node_index, zip(nodes_x, nodes_y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # now add edges
    edge_data = {k: edge_df[v] for k, v in {**edge_data_cols, "start": edge_start, "end": edge_end}.items()}
    graph.edge_renderer.data_source.data = edge_data
    if edge_weight not in edge_df.columns:
        w = np.ones(edge_df.shape[0])
    else:
        w = edge_df[edge_weight]
    if normalize_alpha:
        a = w / np.max(w)
    else:
        a = np.ones_like(w)
    edge_linewidth = np.log(w) + 1
    graph.edge_renderer.data_source.data['line_width'] = edge_linewidth
    graph.edge_renderer.data_source.data['line_alpha'] = a

    # modify the appearance of nodes and edges
    graph.node_renderer.glyph = Circle(radius=sin(2 * pi / n_nodes / 3), **node_properties)
    graph.node_renderer.hover_glyph = Circle(radius=sin(2 * pi / n_nodes / 3), **hover_node_properties)
    graph.node_renderer.selection_glyph = Circle(radius=sin(2 * pi / n_nodes / 3), **selected_node_properties)

    graph.edge_renderer.glyph = MultiLine(**edge_properties)
    graph.edge_renderer.hover_glyph = MultiLine(**hover_edge_properties)
    graph.edge_renderer.selection_glyph = MultiLine(**selected_edge_properties)

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

    tools = [BoxZoomTool(), ResetTool(), TapTool()]
    if add_hovertool:
        node_hover_tool = HoverTool(tooltips=[(k, v) for k, v in hover_tooltips.items()])
        tools.append(node_hover_tool)

    return graph, tools


def circle_layout_graph_v2(node_df, edge_df,
                           node_index_name="id", use_node_df_index=True, node_data_cols=None,
                           fill_color_by=None, fill_color_function=bokeh.palettes.turbo,
                           line_color_by=None, line_color_function=bokeh.palettes.turbo,
                           node_line_width=3, node_properties={},
                           hover_node_properties={"line_width": 3},
                           selected_node_properties={"line_width": 3},
                           edge_start_col="node1", edge_end_col="node2", edge_weight_col=None, edge_data_cols=None,
                           log_width=True, use_alpha=True, default_edge_linewidth=3,
                           edge_properties={},
                           hover_edge_properties={},
                           selected_edge_properties={},
                           layout_origin=np.zeros(2), layout_radius=1.0,
                           circular_arcs=True, circle_k=3,
                           add_hovertool=False, hover_tooltips={}):
    """Return a bokeh.GraphRenderer object that can plot the specified graph.

    `node_data_cols` lists which columns of `node_df` should be passed to the node renderer in a `ColumnDataSource`
    Likewise for `edge_data_cols`. Note that `edge_start_col` and `edge_end_col` should be names of columns in `edge_df`;
    they'll get renamed to 'start' and 'end' before getting passed to the edge renderer.

    If no `edge_weight_col` is passed, all edges will have the same line width (set by `default_linewidth`), and `use_alpha`
    will be ignored. Otherwise, edges will have varying width ."""
    graph = GraphRenderer()

    edge_df = edge_df.rename(columns={edge_start_col: "start", edge_end_col: "end"})
    if node_data_cols is None:
        node_data_cols = node_df.columns.to_list()
    if edge_data_cols is None:
        edge_data_cols = edge_df.columns.to_list()

    # add columns to dataframes as needed
    if "theta" not in node_df.columns:
        theta = np.linspace(0, 2 * np.pi, node_df.shape[0] + 1)[:-1]
        node_df["theta"] = theta
    if "fill_color" not in node_df.columns:
        # here we'll do some fancy categorical mapping
        if fill_color_by is None:
            fill_color_by = node_df.columns.to_list()
        fill_color_mapper = pd.DataFrame(node_df[fill_color_by].value_counts())
        fill_color_mapper["fill_color"] = fill_color_function(fill_color_mapper.shape[0])
        node_df = node_df.reset_index().merge(fill_color_mapper, left_on=fill_color_by, right_index=True).set_index(node_index_name)
        node_data_cols.append("fill_color")
    if "line_color" not in node_df.columns:
        if line_color_by is None:
            line_color_by = node_df.columns.to_list()
        line_color_mapper = pd.DataFrame(node_df[line_color_by].value_counts())
        line_color_mapper["line_color"] = line_color_function(line_color_mapper.shape[0])
        node_df = node_df.reset_index().merge(line_color_mapper, left_on=line_color_by, right_index=True).set_index(node_index_name)
        node_data_cols.append("line_color")

    graph.node_renderer.data_source = ColumnDataSource(node_df[node_data_cols])
    if use_node_df_index:  # overwrites existing index, if defined!
        graph.node_renderer.data_source.data["index"] = node_df.index
    node_index = graph.node_renderer.data_source.data["index"]  # it'll be useful later to have this named
    n_nodes = len(node_index)  # also useful down the road

    # add node layout info
    nodes_x = layout_origin[0] + layout_radius * np.cos(node_df['theta'])
    nodes_y = layout_origin[1] + layout_radius * np.sin(node_df['theta'])
    graph_layout = dict(zip(node_index, zip(nodes_x, nodes_y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # now add edges
    if edge_weight_col is not None:
        #         edge_df = edge_df.rename(columns={edge_weight_col:"weight"})
        if log_width:
            edge_df["line_width"] = np.log(edge_df[edge_weight_col]) + 1
        else:
            edge_df["line_width"] = edge_df[edge_weight_col]
    else:
        edge_df["line_width"] = np.ones(edge_df.shape[0]) * default_edge_linewidth
    edge_data_cols.append("line_width")
    if use_alpha:
        edge_df["alpha"] = edge_df["line_width"] / edge_df["line_width"].max()
    else:
        edge_df["alpha"] = np.ones(edge_df.shape[0])
    if log_width:
        edge_df["line_width"] = np.log(edge_df["line_width"]) + 1
    graph.edge_renderer.data_source = ColumnDataSource(edge_df[edge_data_cols])

    # modify the appearance of nodes and edges
    graph.node_renderer.glyph = Circle(radius=sin(2 * pi / n_nodes / 3), **node_properties)
    graph.node_renderer.hover_glyph = Circle(radius=sin(2 * pi / n_nodes / 3), **hover_node_properties)
    graph.node_renderer.selection_glyph = Circle(radius=sin(2 * pi / n_nodes / 3), **selected_node_properties)

    graph.edge_renderer.glyph = MultiLine(line_width="line_width", **edge_properties)
    graph.edge_renderer.hover_glyph = MultiLine(**hover_edge_properties)
    graph.edge_renderer.selection_glyph = MultiLine(**selected_edge_properties)

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

    tools = [BoxZoomTool(), ResetTool(), TapTool()]
    if add_hovertool:
        node_hover_tool = HoverTool(tooltips=[(k, v) for k, v in hover_tooltips.items()])
        tools.append(node_hover_tool)

    return graph, tools
