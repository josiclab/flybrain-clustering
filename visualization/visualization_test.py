import os
from reduce_graphs import cluster_codes
from vis import circle_layout_graph, breakdown_flowchart_graph, code_heatmap


import pandas as pd
from bokeh.plotting import figure
from bokeh.io import output_file, show
import sys
import logging
from logging import debug, info
logging.basicConfig(format="%(asctime)s  %(module)s> %(levelname)s: %(message)s", datefmt="%Y %m %d %H:%M:%S", level=logging.INFO)

debug("Python version: %s", sys.version_info)


hemibrain_version = "v1.1"
info("Setting up directory information for data set %s", hemibrain_version)
preproc_node_file = "../hemibrain/preprocessed-" + hemibrain_version + "/preprocessed_nodes.csv"
preproc_undirected_edges = "../hemibrain/preprocessed-" + hemibrain_version + "/preprocessed_undirected_edges.csv"
directed_edges = "../hemibrain/exported-traced-adjacencies-" + hemibrain_version + "/traced-total-connections.csv"

list_of_params = ['0', '0.05', '0.1', '0.25', '0.5', '0.75', '1', 'celltype', 'instance']
reneel_params = ['0', '0.05', '0.1', '0.25', '0.5', '0.75', '1']

info("Attempting to read nodes from file %s", preproc_node_file)
FB_node_df = pd.read_csv(preproc_node_file, index_col=0)
info("Loaded full node data")
print(FB_node_df.head())

info("Attempting to read file %s", preproc_undirected_edges)
uFB_edge_df = pd.read_csv(preproc_undirected_edges, index_col=0)
uFB_edge_df.reset_index(drop=True, inplace=True)
info("Loaded full undirected edge data")
print(uFB_edge_df.head())

info("Attempting to read file %s", directed_edges)
FB_edge_df = pd.read_csv(directed_edges).rename(columns={"bodyId_pre": "pre", "bodyId_post": "post", "weight": "total_weight"})
info("Loaded full directed edge data")
print(FB_edge_df.head())

debug("Selecting cluster 0.0/8")
nodes_0_8_df = FB_node_df[FB_node_df["0"] == 8][list_of_params]
edges_0_8_df = uFB_edge_df[(uFB_edge_df['node1'].isin(nodes_0_8_df.index)) & (uFB_edge_df['node2'].isin(nodes_0_8_df.index))]

# Test the circle graph plotter
info("Plotting cluster 0.0/8 using circle_graph")
output_file("../tests/circle_graph.html", title="Circle graph test")

graph, tools = circle_layout_graph(nodes_0_8_df, edges_0_8_df,
                                   node_data_cols=["celltype", "instance"],
                                   node_fill_by=reneel_params, node_line_by=["instance"],
                                   hover_tooltips={"id": "@index", "type": "@celltype", "inst": "@instance"})

plot = figure(title="Cluster 0.0/8", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1), plot_width=800, plot_height=800)
plot.renderers.append(graph)
plot.add_tools(*tools)
info("Successfully plotted %d nodes", len(graph.node_renderer.data_source.data["index"]))
show(plot)

info("Plotting cluster 0.0/8 breakdown figure using breakdown_flowchart_graph")
output_file("../tests/breakdown_graph.html", title="Flowchart graph test")

graph, tools, ranges = breakdown_flowchart_graph(nodes_0_8_df,
                                                 hover_tooltips={"cluster": "@col_value",
                                                                 "param": "@col_name",
                                                                 "size": "@height",
                                                                 "x": "@node_x", "y": "@node_y"})
plot2 = figure(title="Cluster 0.0/8 Breakdown", x_range=ranges[0], y_range=ranges[1], plot_width=800, plot_height=800)
plot2.renderers.append(graph)
plot2.add_tools(*tools),
info("Successfully plotted!")
show(plot2)

info("Plotting cluster code heatmap for cluster 0/8")
output_file("../tests/code_heatmap.html", title="Heatmap test")
cluster_codes = cluster_codes(FB_node_df, FB_edge_df, "0", reset_type=int, property_name="parameter", additional_node_columns=["celltype", "instance"])
plot3 = code_heatmap(cluster_codes[cluster_codes["node"]["0"] == 8], ["pre_count", "post_count"], node_data=["celltype", "instance"], fig_title="Cluster 0/8 code heatmap", color_mapping="log")
info("Successfully plotted!")
show(plot3)
