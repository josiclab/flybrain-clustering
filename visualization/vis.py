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
                          NodesAndLinkedEdges, FixedTicker,
                          HoverTool, TapTool, ColumnDataSource,
                          LinearColorMapper, LogColorMapper, CategoricalColorMapper,
                          ColorBar, BasicTicker, BoxZoomTool,
                          BoxSelectTool, FactorRange)
from bokeh.transform import transform, factor_cmap, linear_cmap, log_cmap

from PIL import Image

from math import sin, pi, sqrt, ceil

reneel_params = list(sorted(['0.0', '1.0', '0.5', '0.25', '0.1', '0.05', '0.75'], key=float))


def draw_graph(node_df, edge_df,
               layout="auto", edge_draw_function=None, loop_draw_function=None,
               edge_start_col="start", edge_end_col="end", edge_weight_col="weight",
               scale_nodes_by=None, node_scale_mode="ones",
               r_min="auto", r_max="auto", r_max_factor=1 / 3, r_min_factor=1 / 3,
               scale_edges_by=None, edge_scale_mode="linear", e_min="auto", e_max="auto",
               scale_loops_by=None, loop_scale_mode="linear", l_min="auto", l_max="auto",
               node_fill_mode=None, node_fill_by="color", node_default_color="black", node_fill_palette=None,
               node_line_mode=None, node_line_by="color", node_line_default_color="black", node_line_width=0,
               edge_color_mode=None, edge_color_by="color", edge_default_color="black",
               loop_color_mode=None, loop_color_by="color", loop_default_color="gray",
               hover_fill_color="#00ff00", hover_line_color="#00aa00",
               selected_fill_color="#ff0000", selected_line_color="#ff0000",
               node_glyph_kwargs={}, edge_glyph_kwargs={},
               loop_r="auto", loop_k=6,
               circle_layout_kwargs={}):
    """
    Return a `bokeh.GraphRenderer` object that displays the graph specified by the node and edge df.

    Parameters:
        node_df:    Specify the nodes. The index of this dataframe will be used as the node index
        edge_df:    The edges
        layout:     "auto"(default), "circle", or a dictionary of `node: (x,y)` key-value pairs. More below.
        edge_draw_function: Function which accepts two points and returns `(xs, ys)`.
                            Default is `None` which results in straight lines
        loop_draw_function: Default is None, to ignore self-loop edges. If a function is passed, use it to draw self-loops
                            See notes below
        edge_start_col, edge_end_col, edge_weight_col: specify columns of `edge_df` to use
        node_fill_mode: How to color the nodes.
                        `None` will use `node_default_color` (default is "black")
                        "categorical" will attempt to build a `factor_cmap`, using the unique values in
                        `node_df[node_fill_by]` and palette `node_fill_palette` (if `None`, use `cc.glasbey`)
                        "custom" will simply specify the fill color as `node_fill_by`, so that column should
                        specify node color
        node_line_mode: Same as `node_fill_mode`, except the accompanying things are `node_line_by` and `node_line_default_color`
        edge_color_mode, loop_color_mode:   Same as `node_fill_mode`. See options below
        scale_nodes_by: If `None`, or if `node_scale_mode="ones"`, all nodes have radius `r_max` (more below).
                        Otherwise, will use the `normalize` function to scale the column `node_df[scale_nodes_by]`
                        to the interval (r_min, r_max)
        node_scale_mode:    See `normalize` for more details, but options are "linear", "log", "sqrt", "ones"(default)
        r_min, r_max:   If either is "auto", will use a heuristic to size the nodes (using minimum distance between two nodes).
                        r_max will be `r_max_factor` (default 1/3) times the minimum distance,
                        r_min will be `r_min_factor` (default 1/3) times r_max
                        Used to apply `normalize` to `scale_nodes_by` column (if specified)
        scale_edges_by: Default is None, to use edge_weight_col. Relevant parameters are `edge_scale_mode` (default "linear"), `e_min`, `e_max`
        scale_loops_by: If `loop_draw_function` is not `None`, this controls the scaling behavior. By default, this is `None`,
                        which will combine all edges (loops and non-loops) and scale all weights together to get edge thickness.
                        If this is not `None`, scale the loops separately, using the specified column.
                        Further parameters are `loop_scale_mode` and `l_min`, `l_max`
        edge_default_color, loop_default_color, etc:    control the color of things
        node_glyph_kwargs: Dictionary with additional keyword args to send to `graph.node_renderer.glyph`
        circle_layout_kwargs: Dictionary with additional kwargs to send to circle_layout(), if layout="circle"

    Graph layout:
        "auto": First check if `node_df` has columns "x","y". If not, throw an error.
                It's a possible goal to implement a variety of graph drawing algorithms, but a lot of them
                are already available in networkx, so there's not much incentive - just get nx to provide
                a dictionary
        "circle": Place the nodes evenly spaced around a unit circle. Sets auto `r_min,r_max` to 0, Dtheta/3
        dict: If a dictionary is passed, use that.

    Edge and loop drawing:
        edge_draw_function must accept two arguments and return a list of xs and list of ys.
        loop_draw_function must also accept two arguments, even though they'll be the same.
        This is to facilitate using the same drawing function for both.

    Color:
        Nodes will be drawn with `node_default_color` and `node_default_line_color` for fill and line color, respectively,
        unless color is specified, using `node_fill_mode`
        "custom" will use the values in `node_df[node_fill_by]` (analogously for line color)
        "categorical" will use a `factor_cmap` with factors defined by `node_df[node_fill_by].unique()` and palette
        specified by `node_fill_palette` (default is `cc.glasbey`)
        Other options may be implemented, but currently those are it.

        Edges currently support three modes, specified by `edge_color_mode`:
        `edge_color_mode=None` will use `edge_default_color` for all edges.
        `edge_color_mode="start"`  will use the same color as the start node
        `edge_color_mode="end"` will use the same color as the end node
    """
    graph = GraphRenderer()

    n_nodes = node_df.shape[0]
    ################################################################################
    # add nodes
    ################################################################################
    graph.node_renderer.data_source.data = node_df
    # in case node_df.index has a name other than "index", add "index"
    graph.node_renderer.data_source.data["index"] = node_df.index

    # Layout
    if layout == "auto":
        graph_layout = dict(zip(node_df.index, zip(node_df["x"], node_df["y"])))
    elif layout == "circle":
        # theta = np.linspace(0, 2 * np.pi, n_nodes + 1)[:-1]
        # xs, ys = np.cos(theta), np.sin(theta)
        # graph_layout = dict(zip(node_df.index, zip(xs, ys)))
        # Dtheta = theta[1] - theta[0]
        graph_layout = circle_layout(node_df, **circle_layout_kwargs)
        Dtheta = 2 * np.pi / n_nodes
        if r_min == "auto":
            r_min = np.sin(Dtheta * r_max_factor * r_min_factor)
        if r_max == "auto":
            r_max = np.sin(Dtheta * r_max_factor)
        if loop_draw_function is not None:  # override the desired loop draw with this one
            if loop_r == "auto":
                loop_r = r_max

            def loop_draw_function(p, q):
                return teardrop(p, p, loop_r, Dtheta, 2 ** loop_k)
                # return outer_loop_circle([0, 0], p, loop_r, k=loop_k)
    else:  # Assume that layout is a dict specifying a layout
        graph_layout = layout
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    if r_min == "auto" or r_max == "auto":
        # compute the closest pair of nodes,
        # scale r_max to some comfortable factor of that,
        # and scale r_min as a factor of r_max
        X = np.array(list(graph_layout.values())).T
        d_min = closest_pair_distance(X)
        if r_max == "auto":
            r_max = d_min * r_max_factor
        if r_min == "auto":
            r_min = r_max * r_min_factor

    # Size
    if node_scale_mode == "custom":
        # the column specified by scale_nodes_by is the raw radius information
        graph.node_renderer.data_source.data["radius"] = node_df[scale_nodes_by]
    else:
        if scale_nodes_by is None:
            radii = np.repeat(r_max, n_nodes)
        else:
            radii = normalize(node_df[scale_nodes_by], r_min, r_max, node_scale_mode)
        graph.node_renderer.data_source.data["radius"] = radii

    ################################################################################
    # style nodes
    ################################################################################
    node_kwargs = {}
    # fill color
    if node_fill_mode == "categorical":
        if node_fill_palette is None:
            node_fill_palette = cc.glasbey
        node_color_mapper = CategoricalColorMapper(factors=node_df[node_fill_by].unique(), palette=node_fill_palette)
        # node_kwargs["fill_color"] = factor_cmap(node_fill_by, node_fill_palette, node_df[node_fill_by].unique())
        node_kwargs["fill_color"] = transform(node_fill_by, node_color_mapper)
    elif node_fill_mode == "custom":
        node_kwargs["fill_color"] = node_fill_by
    else:  # no color mode specified
        node_kwargs["fill_color"] = node_default_color
    # line color
    if node_line_mode == "custom":
        node_kwargs["line_color"] = node_line_by
    else:
        node_kwargs["line_color"] = node_line_default_color
    node_kwargs.update(node_glyph_kwargs)

    graph.node_renderer.glyph = Circle(radius="radius", line_width=node_line_width, **node_kwargs)
    graph.node_renderer.hover_glyph = Circle(radius="radius", line_width=node_line_width, fill_color=hover_fill_color, **node_glyph_kwargs)
    graph.node_renderer.selection_glyph = Circle(radius="radius", line_width=node_line_width, fill_color=selected_fill_color, **node_glyph_kwargs)

    ################################################################################
    # add edges
    ################################################################################
    if scale_edges_by is None:
        scale_edges_by = edge_weight_col

    # split up the edge df into self loops and cross edges
    loops = pd.DataFrame(edge_df[edge_df[edge_start_col] == edge_df[edge_end_col]])
    loops["color"] = loop_default_color
    edges = pd.DataFrame(edge_df[edge_df[edge_start_col] != edge_df[edge_end_col]])
    edges["color"] = edge_default_color
    if e_max == "auto":
        e_max = 40 * r_max  # Let's try this heuristic
    if e_min == "auto":
        e_min = e_max / 20
    if l_max == "auto":
        l_max = e_max
    if l_min == "auto":
        l_min = e_min
    if loop_draw_function is not None:
        # I want to draw loops. How should the width be scaled?
        if scale_loops_by is None:
            # loops should not be scaled separately
            scale_loops_by = scale_edges_by
            edge_df = pd.concat([loops, edges], ignore_index=True)                                  # combine
            edge_df["width"] = normalize(edge_df[scale_edges_by], e_min, e_max, edge_scale_mode)    # then scale
        else:
            # loops should be scaled separately
            loops["width"] = normalize(loops[scale_loops_by], l_min, l_max, loop_scale_mode)    # scale
            edges["width"] = normalize(edges[scale_edges_by], e_min, e_max, edge_scale_mode)
            edge_df = pd.concat([loops, edges], ignore_index=True)                              # then combine
    else:
        # Don't draw loops
        edge_df = edges
        edge_df["width"] = normalize(edge_df[scale_edges_by], e_min, e_max, edge_scale_mode)

    # edge_data = {"start": edge_df[edge_start_col],
    #              "end": edge_df[edge_end_col],
    #              "width": edge_df["width"],
    #              "line_color": edge_df["color"]}
    # graph.edge_renderer.data_source.data = edge_data
    edge_df = edge_df.rename(columns={edge_start_col: "start", edge_end_col: "end"})
    graph.edge_renderer.data_source.data = edge_df

    ################################################################################
    # style edges
    ################################################################################
    edge_kwargs = dict(line_color="color", line_width="width")
    if edge_color_mode in ["start", "end"]:
        if node_fill_mode == "categorical":
            graph.edge_renderer.data_source.data["color"] = node_df.loc[edge_df[edge_color_mode], node_fill_by].values
            edge_kwargs["line_color"] = transform("color", node_color_mapper)
            # edge_kwargs["color"] = factor_cmap(edge_color_mode, node_fill_palette, node_df[node_fill_by].unique())
        elif node_fill_mode == "custom":
            # edge_df["color"] = node_df.loc[edge_df[edge_color_mode], node_fill_by].values
            # update the data source, not the dataframe.
            graph.edge_renderer.data_source.data["color"] = node_df.loc[edge_df[edge_color_mode], node_fill_by].values
            edge_kwargs["line_color"] = "color"
        else:
            edge_kwargs["line_color"] = "color"
    graph.edge_renderer.glyph = MultiLine(**edge_kwargs, **edge_glyph_kwargs)
    graph.edge_renderer.hover_glyph = MultiLine(line_color=hover_line_color, line_width="width", **edge_glyph_kwargs)
    graph.edge_renderer.selection_glyph = MultiLine(line_color=selected_line_color, line_width="width", **edge_glyph_kwargs)
    graph.edge_renderer.nonselection_glyph = MultiLine(**edge_kwargs, line_alpha=0.5, **edge_glyph_kwargs)

    ################################################################################
    # draw edges
    ################################################################################
    if edge_draw_function is not None:
        xs, ys = [], []
        for u, v in zip(graph.edge_renderer.data_source.data["start"], graph.edge_renderer.data_source.data["end"]):
            if u != v:
                x, y = edge_draw_function(graph_layout[u], graph_layout[v])
            else:  # theoretically, you can't reach this point if loop_draw_function is not specified
                if layout == "circle":
                    query = graph.node_renderer.data_source.data["index"] == u
                    node_radius = graph.node_renderer.data_source.data["radius"][query][0]
                    x, y = teardrop(graph_layout[u], graph_layout[u], node_radius, np.pi / 3)
                else:
                    x, y = loop_draw_function(graph_layout[u], graph_layout[v])
            xs.append(x)
            ys.append(y)
        graph.edge_renderer.data_source.data["xs"] = xs
        graph.edge_renderer.data_source.data["ys"] = ys

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    return graph


def circle_layout_graph(node_df, edge_df,
                        node_data_cols=[],
                        node_index_name="id", use_node_df_index=True,
                        node_fill_by="index", node_fill_color_mode=None,
                        node_line_by="index", node_line_color_mode=None, node_line_width=3,
                        scale_nodes_by=None, log_scale_nodes_by=None, r_min=0.0, node_scale_mode="linear",
                        node_properties={},
                        default_line_color="#000000", default_loop_color="#222222",
                        hover_fill_color="#00ff00", hover_line_color="#00aa00",
                        selected_fill_color="#ff0000", selected_line_color="#ff0000",
                        edge_start_col="node1", edge_end_col="node2",
                        edge_weight_col="total_weight", edge_weight_mode="linear",
                        edge_data_cols=[], use_alpha=False, log_weight=True, edge_scale=1.0, e_min=0.0, l_min=0.0,
                        loop_mode=None, loop_scale=1.0, loop_radius_scale=1.0,
                        node_fill_palette="default", node_line_palette="default",
                        layout_origin=np.zeros(2), layout_radius=1.0,
                        circular_arcs=True, circle_k=3,
                        hover_tooltips={"id": "@index"}):
    """
    Return a `bokeh.GraphRenderer` object and list of tools that display the graph specified by the input dataframes.

    Required arguments are `node_df`, `edge_df` which specify the structure of the graph.
    The column `node_df[node_index_name]` or `node.index` will be used as the graph renderer index.
    Other columns can be stored in the graph renderer by specifying a list `node_data_cols` (default empty list `[]`).

    `node_fill_by` specifies a data column to use for colormapping the node fill color.
    `node_fill_color_mode` is one of "categorical", "custom", "linear", "log", or None. For "categorical",
    the data in `node_df[node_fill_by]` must be strings. For "custom", the values in `node_df[node_fill_by]`
    must be colors (e.g. hex string, rgb values, etc). For `log` and `linear`, the minimum and maximum values
    of the column will be sent to a `linear_cmap` or `log_cmap`. Passing `None` makes the nodes gray.
    `node_line_by` and `node_line_color_mode` behave identically.
    By default, `node_fill_by` is "index", which will use whatever index is used for the nodes
    (this should work regardless of what the index column was called, or if the DataFrame index was used,
    as under the hood a new data column is created named "index" to work with Bokeh's GraphRenderer.)

    `loop_mode` determines how loops are drawn. By default, it's `None`, which will ignore edges where start = end.
    Other options:
        * "circle" or "loop" will draw a circle tangent to the layout circle
        * "border" will use the border of the nodes to represent self-edges
    """
    if isinstance(node_fill_palette, str) and node_fill_palette == "default":
        node_fill_palette = cc.glasbey_dark
    if isinstance(node_line_palette, str) and node_line_palette == "default":
        node_line_palette = cc.glasbey_light
    graph = GraphRenderer()

    n_nodes = node_df.shape[0]

    # add edges
    # split up the edge df into self loops and cross edges
    loop_df = pd.DataFrame(edge_df[edge_df[edge_start_col] == edge_df[edge_end_col]])
    loop_df["line_color"] = default_loop_color
    loop_df["width"] = loop_scale * LNL_normalize(loop_df[edge_weight_col], l_min, edge_weight_mode)

    edge_df = pd.DataFrame(edge_df[edge_df[edge_start_col] != edge_df[edge_end_col]])
    edge_df["line_color"] = default_line_color
    edge_df["width"] = edge_scale * LNL_normalize(edge_df[edge_weight_col], e_min, edge_weight_mode)

    # depending on the loop mode, we combine data in various ways
    if loop_mode in ["circle", "loop"]:
        edge_df = pd.concat([edge_df, loop_df], ignore_index=True)
    elif loop_mode == "border":
        merge_args = {"right": loop_df[[edge_start_col, edge_weight_col, "width", "line_color"]],
                      "right_on": edge_start_col,
                      "how": "left"}
        if use_node_df_index:
            merge_args["left_index"] = True
            node_index_name = node_df.index.name
        else:
            merge_args["left_on"] = node_index_name
        node_df = node_df.merge(**merge_args)
        # if use_node_df_index:
        #     node_df = node_df.set_index(node_index_name)
        # if use_node_df_index:
        #     node_df = node_df.set_index(n)
        # node_df.fillna({edge_weight_col: 0.0,
        #                 "width": 0.0,
        #                 "line_color": default_loop_color},
        #                inplace=True)
        node_line_width = "width"  # override the fixed width
    # else: by default, drop the self loops

    edge_data = {"start": edge_df[edge_start_col],
                 "end": edge_df[edge_end_col],
                 "width": edge_df["width"],
                 "line_color": edge_df["line_color"],
                 ** {k: edge_df[k] for k in edge_data_cols}}
    # if use_alpha:
    #     # edge_data["alpha"] = edge_data["weight"] / edge_data["weight"].max()
    #     edge_data["alpha"] = normalized_weights
    # else:
    #     # edge_data["alpha"] = np.ones_like(edge_data["weight"])
    #     edge_data["alpha"] = np.ones_like(normalized_weights)
    edge_data["alpha"] = np.ones_like(edge_data["width"])
    # edge_data["weight"] = edge_data["weight"] * edge_scale
    # edge_data["weight"] = edge_scale * (e_min + (1 - e_min) * normalized_weights)
    # edge_data["weight"] = edge_scale * normalized_weights
    graph.edge_renderer.data_source.data = edge_data

    # Use the node DF as the data source for the node renderer
    # if len(node_data_cols) == 0:
    #     node_data_cols = node_df.columns
    graph.node_renderer.data_source.data = node_df  # [node_data_cols]
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

    # add layout info to the data source, so it can be accessed externally
    graph.node_renderer.data_source.data["theta"] = theta
    graph.node_renderer.data_source.data["x"] = nodes_x
    graph.node_renderer.data_source.data["y"] = nodes_y

    # style the nodes
    # Fill color
    if node_fill_color_mode == "categorical":
        node_fill = factor_cmap(node_fill_by,
                                palette=repeat_to_match_lengths(node_fill_palette, n_nodes),
                                factors=node_df[node_fill_by].unique())
    # elif node_fill_color_mode == "modular":  # does not currently work
    #     graph.node_renderer.data_source.data[node_fill_by] = [node_fill_palette[c % len(node_fill_palette)] for c in graph.node_renderer.data_source.data["index"]]
    #     node_fill = node_fill_by
    elif node_fill_color_mode == "custom":
        # assume that the column node_fill_by is a list of colors
        node_fill = node_fill_by
    elif node_fill_color_mode == "linear" or node_fill_color_mode == "log":
        # not allowed to use index for this one
        # node_min, mode_max = node_df[node_fill_by].min(), node_df[node_fill_by].max()
        mapper_params = {"field_name": node_fill_by,
                         "palette": node_fill_palette,
                         "low": node_df[node_fill_by].min(),
                         "high": node_df[node_fill_by].max()}
        if node_fill_color_mode == "linear":
            node_fill = linear_cmap(**mapper_params)
        else:
            node_fill = log_cmap(**mapper_params)
    else:
        node_fill = "gray"

    # Line color
    if node_line_color_mode == "categorical":
        node_line_color = factor_cmap(node_line_by,
                                      palette=repeat_to_match_lengths(node_line_palette, n_nodes),
                                      factors=node_df[node_line_by].unique())
    elif node_line_color_mode == "custom":
        node_line_color = node_line_by
    elif node_line_color_mode == "linear" or node_line_color_mode == "log":
        mapper_params = {"field_name": node_line_by,
                         "palette": node_line_palette,
                         "low": node_df[node_line_by].min(),
                         "high": node_df[node_line_by].max()}
        if node_line_color_mode == "linear":
            node_line_color = linear_cmap(**mapper_params)
        else:
            node_line_color = log_cmap(**mapper_params)
    else:
        node_line_color = default_loop_color
    if loop_mode == "border":  # override color scheme
        node_line_color = default_loop_color

    # Size
    r_max = sin(2 * pi / n_nodes / 3)
    graph.node_renderer.data_source.data["radius"] = r_max * LNL_normalize(node_df[scale_nodes_by], r_min, node_scale_mode)
    graph.node_renderer.glyph = Circle(radius="radius", fill_color=node_fill, line_color=node_line_color, line_width=node_line_width, **node_properties)
    graph.node_renderer.hover_glyph = Circle(radius="radius", fill_color=hover_fill_color, line_color=hover_line_color, line_width=node_line_width, **node_properties)
    graph.node_renderer.selection_glyph = Circle(radius="radius", fill_color=selected_fill_color, line_color=selected_line_color, line_width=node_line_width, **node_properties)

    # style the edges
    graph.edge_renderer.glyph = MultiLine(line_color="line_color", line_width="width", line_alpha="alpha")
    graph.edge_renderer.hover_glyph = MultiLine(line_color=hover_line_color, line_width="width", line_alpha=1.0)
    graph.edge_renderer.selection_glyph = MultiLine(line_color=selected_line_color, line_width="width", line_alpha=1.0)
    graph.edge_renderer.nonselection_glyph = MultiLine(line_color="line_color", line_alpha=0.5, line_width="width")

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    # Convert edges to circular arcs
    if circular_arcs:
        xs, ys = [], []
        for start_index, end_index in zip(graph.edge_renderer.data_source.data["start"], graph.edge_renderer.data_source.data["end"]):
            try:
                Q = np.array(graph_layout[start_index])
                R = np.array(graph_layout[end_index])
            except Exception as e:
                print("Exception occurred:", e)
                continue
            if start_index != end_index:
                circle_xs, circle_ys = inverted_circle_arc(layout_origin, Q, R, circle_k)
            else:
                circle_xs, circle_ys = outer_loop_circle(layout_origin, Q, loop_radius_scale * r_max, circle_k)
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

    # add the mouseover/click behavior
    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    tools = [TapTool(), HoverTool(tooltips=[(k, v) for k, v in hover_tooltips.items()])]

    suggested_x_range = (min(node_x) - bar_width / 2, max(node_x) + bar_width / 2)
    suggested_y_range = (0, max(y + h / 2 for y, h in zip(node_y, node_height)))

    return graph, tools, (suggested_x_range, suggested_y_range)


def breakdown_barchart_figure(node_df, select_col, select_value,
                              columns=[], title=None,
                              legend=True, palette=None,
                              type_threshold=0, instance_threshold=0,
                              x_coords=None, figsize=(20, 20)):
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
            if palette is not None:
                color = palette[cluster % len(palette)]
            else:
                color = None

            plt.bar([x], [count], bottom=cumulative - count, width=0.8, linewidth=1, edgecolor="black", color="white", label="_nolegend_")
            p = plt.bar([x], [count], bottom=cumulative - count, width=0.8 * fraction, label=leg_ent, color=color)
            if "type" in c and count >= type_threshold:
                celltypes.append(p)
            if c == "instance" and count >= instance_threshold:
                instances.append(p)

    if legend:
        l1 = plt.legend(title="Cell Type", bbox_to_anchor=(1.01, 1), loc='upper left', handles=celltypes, fontsize=12)
        plt.legend(title="Instances", bbox_to_anchor=(1.15, 1), loc="upper left", handles=instances, fontsize=12)
        plt.gca().add_artist(l1)

    plt.xticks(x_coords, columns, fontsize=12)
    plt.xlabel("Parameter")
    plt.ylabel("# Cells in cluster")
    if not title:
        title = "Parameter " + str(select_col) + " Cluster " + str(select_value) + " grouping by algorithm parameters"
    plt.title(title, fontsize=20)

    return f


def code_heatmap(full_df, codes, node_header="node", node_data=[],
                 add_hovertool=False,
                 continuous_palette=cc.fire, category_palette=cc.glasbey,
                 color_mapping="linear",
                 fig_title="Cluster codes",
                 width=800, height=1000,
                 node_font_size="7px", category_font_size="17px"):
    """
    Produces a "code heatmap" from the cluster code dataframe returned by
    `reduce_graphs.cluster_codes`.

    Args:
    :param full_df: Cluster code DataFrame, as returned by `reduce_graphs.cluster_codes`.
    :param codes: Which codes to use (e.g. `["pre_count", "post_count"]`)
    :param node_header: Top-level index for columns with node data
    :param node_data: Which node data columns to include in the plot
        Note that all rows of `full_df` will be used, so pass just the rows you want to see.

    :param add_hovertool: If true, adds a hovertool that displays the values in the heatmap.

    :param continous_palette: Bokeh palette[1] for the heatmap. Default is `colorcet.fire`
    :param category_palette: Bokeh palette for categorical data. Default is `colorcet.glasbey`
    :param color_mapping: One of "linear" (default) or "log". How to map colors for the heatmap.
                          If "log", values will be incremented by one to avoid errors.
    :param fig_title: The title of the Bokeh figure. Default "Cluster codes"
    :param width, height: Size in px of the figure. Default is 800px wide, 1000px tall.
    :param node_font_size, category_font_size: Font sizes for axis labels. Default `7px` and `17px`, respectively.

    :return: A Bokeh `figure`

    [1] A `bokeh` palette is a list of strings, each a hex code for a color; see https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    """
    type_df = full_df[node_header][node_data]
    for c in type_df.columns:
        type_df[c] = type_df[c].astype(str)
    type_df.index = type_df.index.astype(str)
    x_categories = list(type_df.columns)
    if len(node_data) > 0:
        y_categories = type_df.sort_values(by=node_data).index.to_list()
    else:
        y_categories = list(full_df.index.astype(str))
    type_stack = pd.DataFrame(type_df.stack(), columns=["type"]).reset_index()
    type_stack.columns = pd.Series(["id", "col", "value"])
    type_source = ColumnDataSource(type_stack)

    code_df = full_df[codes]
    code_df.index = code_df.index.astype(str)
    code_df.columns = pd.Series([tuple_to_string(c) for c in code_df.columns], name="col")
    x_categories += list(code_df.columns)
    code_stack = pd.DataFrame(code_df.stack(), columns=["code"]).reset_index()
    code_stack.columns = pd.Series(["id", "col", "value"])
    code_source = ColumnDataSource(code_stack)

    if color_mapping == "log":
        code_stack["value"] = code_stack["value"] + 1
        mapper = LogColorMapper(palette=continuous_palette, low=code_stack["value"].min(), high=code_stack["value"].max())
    else:
        mapper = LinearColorMapper(palette=continuous_palette, low=code_stack["value"].min(), high=code_stack["value"].max())

    p = figure(title=fig_title,
               plot_width=width, plot_height=height,
               x_range=x_categories, y_range=y_categories,
               x_axis_location="above",
               tools="ypan,ywheel_zoom,yzoom_in,yzoom_out,save,reset")
    p.add_tools(BoxZoomTool(dimensions="height"))
    p.rect(x="col", y="id",
           width=1, height=1,
           source=code_source,
           line_color=None,
           fill_color=transform("value", mapper))
    category_palette = repeat_to_match_lengths(category_palette, len(type_stack["value"].unique()))
    p.rect(x="col", y="id", width=1, height=1, source=type_source, line_color=None,
           fill_color=factor_cmap("value", palette=category_palette, factors=list(type_stack["value"].unique())))

    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=10))
    p.add_layout(color_bar, "right")

    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.yaxis.major_label_text_font_size = node_font_size
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    p.xaxis.major_label_text_font_size = category_font_size

    if add_hovertool:
        p.add_tools(HoverTool(tooltips=[("id", "@id"), ("info", "@col"), ("value", "@value")]))

    return p


def display_dataframe(df, categorical_columns=None, continuous_columns=None, width_hack=None,
                      name_mapper=str, color_mapping="linear",
                      categorical_palette=cc.glasbey_dark, continuous_palette=cc.fire, nan_color=None,
                      add_hovertool=False, extra_tooltips=[], add_colorbar=True,
                      fig_title="DataFrame Visualization",
                      width=800, height=1000, bg_color="black",
                      y_font_size="7pt", x_font_size="17px", x_orientation=1.0,):
    """
    Display a dataframe, applying a categorical coloring to the columns listed
    in `categorical_columns` and a heatmap to the columns listed in
    `continuous_columns`

    Args:
    :param df: The pandas DataFrame to display
    :param categorical_columns: Which columns to display with a categorical
        coloring. These are all drawn on the left side of the plot.
    :param continuous_columns: Which columns to display with a heatmap.
    :param name_mapper: Function that converts df column names into strings
        to display in the plot
    :param color_mapping: "log" or "linear" (default) for continuous columns
    :param categorical_palette: bokeh palette for categories. Default is `colorcet.glasbey_dark`
    :param continuous_palette: As for categories. Default is `colorcet.fire`
    :param add_hovertool: Default False. Displays details of each entry in the dataframe.
    :param add_colorbar: Default True. Colorbar for the continuous heatmap.
    :param fig_title: The figure title
    :param width, height: The width and height of the plot, in px
    :param bg_color: plot background color
    :param y_font_size: Font size of the y-axis labels (df index)
    :param x_font_size: Font size of the x-axis labels (df columns)
    :param x_orientation: Orientation, in radians, of x-axis labels. Default 1.0
    """
#     y_categories = df.index.astype(str).to_list()
    y_categories = [str(i) for i in df.index]
    x_categories = []

    if categorical_columns is not None:
        # cat_df = df[categorical_columns]
        #         cat_df = pd.DataFrame(index=df.index.astype(str))
        cat_df = pd.DataFrame(index=y_categories)
        for c in categorical_columns:
            cat_df[name_mapper(c)] = df[c].values.astype(str)
        cat_stack = pd.DataFrame(cat_df.stack()).reset_index()
        cat_stack.columns = pd.Series(["id", "col", "value"])
        cat_source = ColumnDataSource(cat_stack)
        x_categories += list(cat_df.columns)

        cat_palette = repeat_to_match_lengths(categorical_palette, len(cat_stack["value"].unique()))
        cat_cmap = factor_cmap("value", palette=cat_palette, factors=list(cat_stack["value"].unique()))

    if continuous_columns is not None:
        #         con_df = pd.DataFrame(index=df.index.astype(str))
        con_df = pd.DataFrame(index=y_categories)
        for c in continuous_columns:
            con_df[name_mapper(c)] = df[c].values
        con_stack = pd.DataFrame(con_df.stack()).reset_index()
        con_stack.columns = pd.Series(["id", "col", "value"])
        con_stack.dropna(inplace=True)
        con_source = ColumnDataSource(con_stack)
        x_categories += list(con_df.columns)

        if color_mapping == "log":
            con_cmap = LogColorMapper(palette=continuous_palette, low=max(con_stack["value"].min(), 1), high=con_stack["value"].max(), nan_color=nan_color)
        else:
            con_cmap = LinearColorMapper(palette=continuous_palette, low=con_stack["value"].min(), high=con_stack["value"].max(), nan_color=nan_color)

    if width_hack is not None:
        width_df = pd.DataFrame(index=y_categories)
        for c in continuous_columns:
            width_df[name_mapper(c)] = df[c.replace(*width_hack)].values
        width_stack = pd.DataFrame(width_df.stack()).reset_index()
        width_stack.columns = pd.Series(['id', 'col', 'width'])
        con_stack = con_stack.merge(width_stack, on=['id', 'col'], how='left')
        con_source = ColumnDataSource(con_stack)
        w = 'width'
    else:
        w = 1

    p = figure(title=fig_title, plot_width=width, plot_height=height,
               x_range=x_categories, y_range=y_categories,
               x_axis_location="above",
               tools="ypan,ywheel_pan,yzoom_in,yzoom_out,save,reset,undo,redo")
    p.add_tools(BoxZoomTool(dimensions="height"))
    if categorical_columns is not None:
        p.rect(x="col", y="id", width=1, height=1, source=cat_source, line_color=None, fill_color=cat_cmap)
    if continuous_columns is not None:
        p.rect(x="col", y="id", width=w, height=1, source=con_source, line_color=None, fill_color=transform("value", con_cmap))
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.yaxis.major_label_text_font_size = y_font_size
    p.xaxis.major_label_text_font_size = x_font_size
    p.xaxis.major_label_orientation = x_orientation
    p.background_fill_color = bg_color

    if add_hovertool:
        p.add_tools(HoverTool(tooltips=[("id", "@id"), ("info", "@col"), ("value", "@value")] + extra_tooltips))

    if add_colorbar:
        color_bar = ColorBar(color_mapper=con_cmap, ticker=BasicTicker(desired_num_ticks=10))
        p.add_layout(color_bar, "right")

    return p


def roi_adj_fig(df, rois,
                suffixes={'width': '_syn_fraction', 'fill_color': '_io_ratio'},
                cat_cols=['type_group', 'celltype', 'instance'] + reneel_params, cat_palette='default',
                color_mapping='linear', con_palette='default', con_cmap_low=-1.0, con_cmap_high=1.0, nan_color='gray',
                x_range_mapper=str, y_range_mapper=str,
                fig_title='Neuron Input/Output per region', width=2000, height=1600, bg_color="white",
                y_font_size="7pt", x_font_size="9pt", x_orientation='vertical',
                extra_tooltips=[]):
    """Specialized version of display_dataframe for displaying neurons input/output in various ROIs.

    Suggested x_range_mapper is `lambda s: pred_tuple.get(s, ('', s))` where `pred_tuple` maps an ROI
    to the tuple representing its incoming edge in the ROI hierarchy.
    """
    if cat_palette == 'default':
        cat_palette = cc.glasbey_light + cc.glasbey
    if con_palette == 'default':
        con_palette = cc.CET_D8

    y_range = FactorRange(factors=[y_range_mapper(y) for y in df.index], bounds='auto')
    x_range = FactorRange(factors=[x_range_mapper(x) for x in cat_cols + rois], bounds='auto')

    cat_stack = pd.DataFrame(df[cat_cols].stack()).reset_index().rename(columns={'level_0': 'id', 'level_1': 'col', 0: 'val'})
    cat_stack['val'] = cat_stack['val'].apply(str)  # for categorical color mapper

    color_stack = pd.DataFrame(df[[c + suffixes['fill_color'] for c in rois]].stack())
    color_stack = color_stack.reset_index().rename(columns={'level_0': 'id', 'level_1': 'col', 0: 'val'})
    color_stack['col'] = color_stack['col'].apply(lambda s: s.replace(suffixes['fill_color'], ''))
    width_stack = pd.DataFrame(df[[c + suffixes['width'] for c in rois]].stack()).reset_index().rename(columns={'level_0': 'id', 'level_1': 'col', 0: 'width'})
    width_stack['col'] = width_stack['col'].apply(lambda s: s.replace(suffixes['width'], ''))
    con_stack = color_stack.merge(width_stack, on=['id', 'col'], how='left')

    if con_cmap_low == 'min':
        con_cmap_low = con_stack['val'].min()
    if con_cmap_high == 'max':
        con_cmap_high = con_stack['val'].max()

    cat_stack['id'] = [y_range_mapper(y) for y in cat_stack['id']]
    cat_stack['col'] = [x_range_mapper(x) for x in cat_stack['col']]
    con_stack['id'] = [y_range_mapper(y) for y in con_stack['id']]
    con_stack['col'] = [x_range_mapper(x) for x in con_stack['col']]

    cat_palette = repeat_to_match_lengths(cat_palette, len(cat_stack['val'].unique()))
    cat_cmap = factor_cmap('val', palette=cat_palette, factors=cat_stack['val'].unique())
    if color_mapping == "log":
        con_cmap = LogColorMapper(palette=con_palette, low=con_cmap_low, high=con_cmap_high, nan_color=nan_color)
    else:
        con_cmap = LinearColorMapper(palette=con_palette, low=con_cmap_low, high=con_cmap_high, nan_color=nan_color)

    p = figure(title=fig_title, plot_width=width, plot_height=height,
               x_range=x_range, y_range=y_range, x_axis_location="above",
               tools="ypan,ywheel_pan,box_zoom,yzoom_in,yzoom_out,save,reset,undo,redo")
    p.add_tools(BoxZoomTool(dimensions="height"))
    p.rect(x='col', y='id', width=1, height=1, source=cat_stack, line_color=None, fill_color=cat_cmap)
    p.rect(x='col', y='id', width='width', height=1, source=con_stack, line_color=None, fill_color=transform('val', con_cmap))

    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.yaxis.major_label_standoff = 0
    p.yaxis.major_label_text_font_size = y_font_size
    p.ygrid.grid_line_color = None
    p.xaxis.major_label_text_font_size = x_font_size
    p.xaxis.major_label_orientation = x_orientation
    p.background_fill_color = bg_color

    p.add_tools(HoverTool(tooltips=[("id", "@id"), ("info", "@col"), ("value", "@val")] + extra_tooltips),
                BoxSelectTool(dimensions='height'))
    colorbar = ColorBar(color_mapper=LinearColorMapper(palette=con_palette, low=-1.0, high=1.0),
                        ticker=FixedTicker(ticks=[-1, 0, 1]),
                        width=200, height=10, orientation='horizontal', location=(0, 0),
                        major_label_overrides={-1: "Input", 0: "Mixed", 1: "Output"},
                        major_tick_in=0)
    p.add_layout(colorbar, 'above')

    return p


def display_edge_list_as_matrix(edge_df, u_col, v_col, wt_col=None,
                                scale="log",
                                color_col=None, color=None, color_scale="log", palette=cc.bkr,
                                x_range=None, y_range=None, x_above=True,
                                x_label=None, y_label=None,
                                fig_title="Directed Adjacency Matrix",
                                x_orientation="horizontal", y_orientation="vertical",
                                width=800, height=800):
    """
    Given a dataframe whose rows correspond to edges in a graph, display an
    adjacency boxplot kind of thing.

    Will always plot `u_col` on x axis and `v_col` on y axis. In the typical
    case, where `u_col` is the source of a directed edge, this will result in
    the rows being "post-synaptic" and columns being "pre-synaptic".
    """
    if x_range is None:
        x_range = edge_df[u_col].unique().astype(str)
    if y_range is None:
        if x_above:
            y_range = list(reversed(edge_df[v_col].unique().astype(str)))
        else:
            y_range = edge_df[v_col].unique().astype(str)

    edge_source = ColumnDataSource(edge_df)
    edge_source.data[u_col] = [str(s) for s in edge_source.data[u_col]]
    edge_source.data[v_col] = [str(s) for s in edge_source.data[v_col]]
    if wt_col is None:
        # wt_col = "weight"
        edge_source.data["size"] = np.ones(edge_df.shape[0])
    else:
        if scale == "log":
            edge_source.data["size"] = np.log(edge_df[wt_col]) / np.log(edge_df[wt_col].max())
        elif scale == "sqrt":
            edge_source.data["size"] = np.sqrt(edge_df[wt_col] / edge_df[wt_col].max())
        else:
            edge_source.data["size"] = edge_df[wt_col] / edge_df[wt_col].max()

    if color_col is not None:
        if color_scale == "log":
            mapper = LogColorMapper(palette=palette, low=edge_df[color_col].min(), high=edge_df[color_col].max())
        else:
            mapper = LinearColorMapper(palette=palette, low=edge_df[color_col].min(), high=edge_df[color_col].max())
        fill_color = transform(color_col, mapper)
    else:
        fill_color = "#1F77B4"
    p = figure(title=fig_title, plot_width=width, plot_height=height,
               x_range=FactorRange(*x_range), y_range=FactorRange(*y_range),
               x_axis_location="above")
    p.rect(x=v_col, y=u_col, width="size", height="size",
           fill_color=fill_color, source=edge_source, line_color=None)
    p.xaxis.axis_label = x_label
    p.xaxis.major_label_orientation = x_orientation
    p.yaxis.axis_label = y_label
    p.yaxis.major_label_orientation = y_orientation
    return p


################################################################################
# Functions for producing graph layouts
################################################################################


def circle_layout(nodes, c=[0, 0], r=1.0, theta_offset=0.0):
    """Given a node dataframe (i.e. treating the index of the dataframe as the nodes of a graph)
    return a dict of node: (x,y) pairs that puts them on a circle centered at c with radius r"""
    if nodes.shape[0] == 1:
        return {nodes.index[0]: (c[0], c[1])}
    c = np.array(c)
    thetas = np.linspace(0, 2 * np.pi, nodes.shape[0] + 1)[:-1] + theta_offset
    return dict(zip(nodes.index, zip(c[0] + r * np.cos(thetas), c[1] + r * np.sin(thetas))))


def circle_groups_layout(nodes, group_col, count_col="instance"):
    """Given a node dataframe, return a graph layout (dict of node: (x,y) pairs)
    such that nodes are grouped by group_col; each group is a circle layout graph
    and the groups are placed around a circle (similar to how the circle layout
    of the reduced graph would look).

    The `count_col` argument specifies a column of `nodes` which will be used with the "count" aggregator. """
    node_index = nodes.index.name
    if not node_index:
        node_index = "index"
    grouped_nodes = nodes.reset_index().groupby(group_col).agg({count_col: "count", node_index: "unique"})
    n_groups = grouped_nodes.shape[0]
    n_max = grouped_nodes[count_col].max()

    group_thetas = np.linspace(0, 2 * np.pi, n_groups + 1)[:-1]
    grouped_nodes["g_x"] = np.cos(group_thetas)
    grouped_nodes["g_y"] = np.sin(group_thetas)

    d_theta_group = 2 * np.pi / n_groups  # arc distance between group centers
    d_theta_node = 2 * np.pi / n_max  # smallest arclength between node centers
    r_g = np.sin(d_theta_group / 3) / (1 + np.sin(d_theta_node / 3))  # radius of the circle around which nodes get placed

    # r_node = r_g * np.sin(d_theta_node / 3)  # max radius of a node

    grouped_nodes["node_thetas"] = grouped_nodes[node_index].apply(lambda ids: np.linspace(0, 2 * np.pi, len(ids) + 1)[:-1])
    grouped_nodes["node_x"] = [r_g * np.cos(thetas) * (thetas.shape[0] > 1) + np.array([gx]) for thetas, gx in zip(grouped_nodes["node_thetas"], grouped_nodes["g_x"])]
    grouped_nodes["node_y"] = [r_g * np.sin(thetas) * (thetas.shape[0] > 1) + np.array([gy]) for thetas, gy in zip(grouped_nodes["node_thetas"], grouped_nodes["g_y"])]
    node_list = [int(v) for v in np.hstack(grouped_nodes[node_index])]  # some weird compatibility issue with bokeh
    layout = dict(zip(node_list, zip(np.hstack(grouped_nodes["node_x"]), np.hstack(grouped_nodes["node_y"]))))
    return layout

################################################################################
# Functions for drawing curves of various sorts
################################################################################


def circle_arc(P, Q, R, k, theta_tol=1 - 1e-8):
    """Given a circle centered at `P` with points `Q` and `R` on the circle,
    returns the arc of the circle from Q to R. Produces `2 ** k` points.

    `P`, `Q`, `R` should be numpy arrays with 2 elements.

    Assumes that the user has checked that indeed `Q` and `R` lie on the same circle centered at `P`.

    If cos(theta) > theta_tol (where theta is the angle between PQ and PR), return a straight line segment."""
    P, Q, R = map(np.array, [P, Q, R])  # avoid headaches
    PQ, PR = Q - P, R - P
    costheta = PQ.dot(PR) / np.linalg.norm(PQ) / np.linalg.norm(PR)
    costheta = min(max(-1, costheta), 1)  # fix numerical errors
    if costheta > theta_tol:
        return np.array([Q[0], R[0]]), np.array([Q[1], R[1]])
    if costheta == -1:
        # strange things happen when costheta is exactly -1, i.e. we're drawing a semicircle.
        # So let's handle that case separately
        r = np.linalg.norm(PQ)
        uPQ = PQ / r
        th = np.arccos(uPQ[0])
        if uPQ[1] < 0:
            th = 2 * np.pi - th
        thetas = np.linspace(th, th + np.pi, 2**k)
        return r * np.cos(thetas) + P[0], r * np.sin(thetas) + P[1]

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


def approximate_circle_arc(c, p0, v0, theta, n_steps=100):
    """Draw an approximate circle arc with a given center, initial point, and initial velocity.
    Draw until an arc of length theta has been drawn.
    This is approximate in that it never computes sin or cosine;
    instead, it relies on theta ~ sin(theta) when theta is small."""
    c, p0, v0 = map(np.array, [c, p0, v0])
    a0 = c - p0  # initial acceleration is toward the center
    sgn = np.sign(a0[0] * v0[1] - a0[1] * v0[0])  # rotate pi/2 clockwise or ccw to get from acc to vel
    rot = np.array([[0, -sgn], [sgn, 0]])
    v0 = rot @ a0
    P = np.zeros((2, n_steps + 1))
    P[:, 0] = p0
    dt = theta / n_steps  # take small steps so sin(dt) ~ dt
    v = np.array(v0)
    a = np.array(a0)
    for i in range(1, n_steps + 1):
        P[:, i] = P[:, i - 1] + dt * v + (1 - np.sqrt(1 - dt ** 2)) * a
        a = c - P[:, i]
        v = rot @ a
    return P[0], P[1]


def normal_intersection(p, q):
    """Given two points `p`, `q`, find the point of intersection of the lines
    normal to p and q (that is, the lines perpendicular to the vectors p, q
    that pass through the points p, q)

    Note that `p` and `q` must be numpy arrays"""
    # special case: if either p or q is at the origin
    if p @ p == 0 or q @ q == 0:
        raise Exception("Normal line to zero vector is undefined")
    # the normal line to p is the intersection of the plane normal to
    # [p0, p1, p2] and the plane z = 1
    p2 = -(p @ p)
    q2 = -(q @ q)
    # so we compute the cross product and scale by the last entry
    # to get a vector [x, y, 1] whose first two coordinates
    # are the point we want.
    k = p[0] * q[1] - p[1] * q[0]
    return np.array([(p[1] * q2 - p2 * q[1]) / k, -(p[0] * q2 - p2 * q[0]) / k])


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


def poincare_geodesic(C, r, P, Q, k=3, diag_tol=1e-8):
    """Given a circle centered at C with radius r, compute the
    hyperbolic geometry geodesic between P and Q.

    If the area of triangle CPQ is less than diag_tol,
    return a straight line segment
    """
    C, P, Q = map(np.array, [C, P, Q])
    # special cases
    S = np.array([[C[0], P[0], Q[0], C[0]], [C[1], P[1], Q[1], C[1]]])
    a = np.abs(np.sum(S[0, :-1] * S[1, 1:] - S[0, 1:] * S[1, :-1])) / 2
    if a < diag_tol:
        return np.array([P[0], Q[0]]), np.array([P[1], Q[1]])

    # general case: P and Q are not collinear with C
    Pp = C + (P - C) * (r ** 2) / ((P - C) ** 2).sum()
    Qp = C + (Q - C) * (r ** 2) / ((Q - C) ** 2).sum()
    Cp = C + normal_intersection((P + Pp) / 2 - C, (Qp + Q) / 2 - C)
    return circle_arc(Cp, P, Q, k)


def low_k_arc(p, q, offset=1, k=5):
    """Draw an arc of a circle from p to q with somewhat low curvature.
    The center of the circle lies on the perpendicular bisector of line segment pq,
    a distance length(pq) * offset from the midpoint."""
    p, q = np.array(p), np.array(q)

    c = (p + q) / 2
    if offset:
        pq = q - p
        d = np.linalg.norm(pq)
        u = pq / d
        v = np.array([-u[1], u[0]])
        c = c + offset * d * v
    return circle_arc(c, p, q, k)


def outer_loop_circle(C, P, r_s, k=4):
    """Given a point P on a circle centered on C, draw a second circle outside and tangent
    to the first at P. The radius of the circle is r_s.
    The drawn circle will have 2 ** k points."""
    C, P = map(np.array, [C, P])
    CP = P - C
    w = CP / np.linalg.norm(CP)
    Pp = P + r_s * w
    u = -r_s * w
    v = np.array([-u[1], u[0]])

    points = Pp[:, None] + np.outer(u, np.cos(np.linspace(0, 2 * np.pi, 2 ** k))) + np.outer(v, np.sin(np.linspace(0, 2 * np.pi, 2 ** k)))
    return points[0], points[1]


def teardrop(c, v, r, a, n_steps=50):
    """Draw a teardrop shape: From c, draw two straight segments of length r at angles of +/-a/2 from v.
    Then draw a circle arc connecting their endpoints.

    Parameters:
        c : the apex of the drop shape (numpy array or something that can be cast to it of shape (2,))
        v : the direction (same format as c; if this is (0,0) throw an error)
        r : the radius (positive scalar)
        a : 0 < a < pi
    """
    if r <= 0:
        raise Exception("Radius must be positive")
    if not 0 < a < np.pi:
        raise Exception("Angle must be positive and less than pi")
    if v[0] == v[1] == 0:
        raise Exception("Direction must be nonzero")
    c, v = np.array(c), np.array(v)
    u = r * v / np.linalg.norm(v)
    rotp = np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]])
    rotm = np.array([[np.cos(-a / 2), -np.sin(-a / 2)], [np.sin(-a / 2), np.cos(-a / 2)]])
    up, um = rotp @ u, rotm @ u
    cp = c + normal_intersection(up, um)
    xc, yc = approximate_circle_arc(cp, c + up, up, np.pi + a, n_steps)
    xs = np.hstack([[c[0]], xc, [c[0]]])
    ys = np.hstack([[c[1]], yc, [c[1]]])
    return xs, ys


def circle_loop(p, q, theta=0.0, r=1, k=5, stop_theta=2 * np.pi):
    """Draw a circle with the following properties: The center of the circle, c, satisfies
    (c - p) forms an angle of theta from the horizontal. The radius of the circle is r.
    The circle arc will start at p and fill out an angle of stop_theta. So, for example,
    if you want to draw a loop which does not quite close a circle, you could use
    stop_theta = 2 * np.pi - 0.1

    Due a q quirk of my graph drawing function, this needs to take two args, so q is ignored."""
    p = np.array(p)
    u = np.array([np.cos(theta), np.sin(theta)])
    c = p + r * u
    thetas = np.linspace(0, 2 * np.pi, 2 ** k) + np.pi + theta
    return c[0] + r * np.cos(thetas), c[1] + r * np.sin(thetas)


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

################################################################################
# Miscellaneous utility functions
################################################################################


def LNL_normalize(arr, b, nl="linear"):
    """Apply a nonlinearity (if `nl` is specified), then linearly scale arr to values in [b, 1.0]
    If an invalid value of `b` is passed (i.e. b < 0 or b >= 1), b is set to 0.

    Valid choices for `nl` are
    "linear": no nonlinearity is applied. Default if invalid value of `nl` is passed.
    "log", "sqrt": Apply this transformation before linear scaling
    "ones": Ignore values and return an array of 1s. """
    if not 0 <= b < 1:
        b = 0
    if len(arr) == 0:
        return np.array([])

    if nl == "log":
        s_arr = np.log(arr)
    elif nl == "sqrt":
        s_arr = np.sqrt(arr)
    elif nl == "ones":
        s_arr = np.ones_like(arr)
    else:
        s_arr = np.array(arr)
    return b + (1 - b) * s_arr / s_arr.max()


def normalize(arr, a=0.0, b=1.0, nl="linear"):
    """Apply a nonlinearity (if `nl` is specified), then scale and translate the values in `arr`
    to between `a` and `b`

    Note that this implicitly assumes that the values in `arr`, after `nl` is applied,
    are nonnegative. The full algorithm is:
    1. apply nonlinearity
    2. scale by (b - a) / max, so the right endpoint is b - a
    3. translate by a, so that right endpoint is b
    Strictly speaking, it's possible for some values to end up less than `a`,
    for example if there are values less than 1 in `arr` and you use `nl="log"`.

    Valid choices for `nl` are
    "linear": no nonlinearity is applied. Default if invalid value of `nl` is passed.
    "log", "sqrt": Apply this transformation before linear scaling
    "ones": Ignore values and return an array of `b`s"""
    if len(arr) == 0:
        return np.array([])

    if nl == "log":
        s_arr = np.log(arr)
    elif nl == "sqrt":
        s_arr = np.sqrt(arr)
    elif nl == "ones":
        return b * np.ones_like(arr)
    else:
        s_arr = np.array(arr)
    return a + (b - a) * s_arr / s_arr.max()


def closest_pair_distance(X):
    """Given a 2 x n array X representing n points in the plane,
    find the closest pair and return their distance.
    Currently uses an inefficient method, computing all pairwise distances with some matrix algebra."""
    XY = X.T @ X  # inner product of all pairs
    XX = np.diag(XY)  # norms of each
    D = XX[:, None] - 2 * XY + XX[:, None].T  # broadcast things to the right shape
    return np.sort(np.sqrt(D))[:, 1].min()


def index_to_unique_list(index, name):
    """
    Convert the given pandas index into a list of strings by concatenating with the given name.
    """
    return [str(name) + str(idx) for idx in index]


def repeat_to_match_lengths(list_to_repeat, length_to_match):
    """Return `list_to_repeat * n`, truncated to length `length_to_match`
    This function is actually unnecessary, see `itertools.cycle`"""
    list_length = len(list_to_repeat)
    n = int(ceil(length_to_match / list_length))
    return (list(list_to_repeat) * n)[:length_to_match]


def tuple_to_string(t):
    """
    Given a multi-index tuple, shortens it for the heatmap plot.
    """
    if "weight" in t[0]:
        marker = "wt"
    elif "count" in t[0]:
        marker = "#"
    else:
        marker = ""
    if "pre" in t[0]:
        return " ".join([marker, "from", str(t[1])])
    elif "post" in t[0]:
        return " ".join([marker, "to", str(t[1])])
    else:
        return marker + str(t[1])


def bbox(im):
    """Get the bounding box of am image, assuming that white pixels are to be ignored"""
    l, u, r, b = im.size[0], im.size[1], 0, 0
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            if sum(im.getpixel((x, y))) < 1020:
                l = min(x, l)
                r = max(x, r)
                u = min(y, u)
                b = max(y, b)
    return l, u, r, b


def simple_crop(fname, add_name="_cropped", display_cropped=False):
    ext = fname.split(".")[-1]
    im = Image.open(fname)
    imbox = bbox(im)
    im = im.crop(imbox)
    im.save(fname.replace("." + ext, add_name + "." + ext))
    if display_cropped:
        display(im)


def pie_chart_angles(arr):
    """Given an array of nonnegative values, return the start and stop angles for a piechart"""
    arr_ = np.array(arr)
    thetas = 2 * np. pi * arr_ / arr_.sum()
    thetas = np.concatenate([[0], thetas]).cumsum()
    return thetas[:-1], thetas[1:]
