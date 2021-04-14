import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import numpy as np
import logging
from logging import debug, info, warning, critical
from bokeh.plotting import figure
from bokeh.io import output_file, show, save


from visualization.vis import breakdown_barchart_figure, breakdown_flowchart_graph

parser = argparse.ArgumentParser()
parser.add_argument('hemibrain_version', default="1.1", help="Hemibrain version, valid as either '1.x' or 'v1.x' format")
parser.add_argument('--fig_dir', default='figures', help="Directory for figures")
parser.add_argument('--fig_fmt', default='png', help="Figure format")
parser.add_argument('--loglevel', choices=["debug", "info", "warning", "error", "critical"], default="info", help="Level of verbosity")

args = parser.parse_args()

# configure logging
loglevel = args.loglevel.upper()
logging.basicConfig(format="%(asctime)s  %(module)s> %(levelname)s: %(message)s", datefmt="%Y %m %d %H:%M:%S", level=getattr(logging, loglevel.upper()))

fig_fmt = args.fig_fmt

hemibrain_version = args.hemibrain_version
list_of_params = ['0.0', '0.05', '0.1', '0.25', '0.5', '0.75', '1.0', 'celltype', 'instance']
reneel_params = ['0.0', '0.05', '0.1', '0.25', '0.5', '0.75', '1.0']
type_params = ['celltype', 'instance']

if "v" not in hemibrain_version:
    hemibrain_version = "v" + hemibrain_version
info("Processing hemibrain data %s", hemibrain_version)

fig_dir = os.path.join(args.fig_dir, hemibrain_version)
if not os.path.exists(fig_dir):
    warning("Figure directory does not exist, attempting to create directory")
    try:
        os.makedirs(fig_dir)
    except Exception as e:
        critical("Error creating directory! Exception arose:")
        print(e)
        exit(1)

preprocessed_nodes_csv = os.path.join("hemibrain/preprocessed-" + hemibrain_version, "preprocessed_nodes.csv")
weighted_edges_csv = os.path.join("hemibrain/exported-traced-adjacencies-" + hemibrain_version, "traced-total-connections.csv")

info("Reading node data from %s", preprocessed_nodes_csv)
FB_node_df = pd.read_csv(preprocessed_nodes_csv)
info("Found these columns:")
print(FB_node_df.columns)

################################################################################
# Cluster size distribution figure
################################################################################
info("Preparing plot of cluster size distribution")
x_coords = list(range(len(reneel_params))) + [len(reneel_params) + 0.5 + i for i in range(len(type_params))]
# info("Plotting things at x_coords = %s", str(x_coords))

f = plt.figure(figsize=(16, 12))

for x, chi in zip(x_coords, list_of_params):
    cluster_sizes = FB_node_df[chi].value_counts().values
    xs = x + 0.8 * (np.random.rand(len(cluster_sizes)) - 0.5)
    plt.scatter(xs, cluster_sizes)

plt.axvline(len(reneel_params) - 0.25, color="gray", linestyle="--")

plt.yscale("log")
plt.ylim(1, 1e4)
plt.yticks(fontsize=14)
plt.ylabel("Cluster Size", fontsize=17)

plt.xticks(x_coords, list_of_params, fontsize=14)
plt.xlabel("Clustering parameter", fontsize=17)

plt.title("Undirected Fly Brain Clustering (Hemibrain " + hemibrain_version + ")", fontsize=22)

f.savefig(os.path.join(fig_dir, "cluster_distribution." + fig_fmt), bbox_inches="tight", format=fig_fmt)
info("Saved cluster size distribution figure to %s", os.path.join(fig_dir, "cluster_distribution.png"))

################################################################################
# Cluster fraction regrouping and flowcharts
################################################################################
info("Preparing to plot cluster breakdowns and flowcharts")
for c in FB_node_df['0.0'].unique():
    if FB_node_df[FB_node_df['0.0'] == c].shape[0] > 100:
        threshold = 10
    else:
        threshold = 0
    f = breakdown_barchart_figure(FB_node_df, '0.0', c, columns=list_of_params, x_coords=[0, 1, 2, 3, 4, 5, 6, 7.5, 8.5], figsize=(16, 16), legend_threshold=threshold)
    plt.axvline(6.75, linestyle="--", color="gray")
    f.savefig(os.path.join(fig_dir, "cluster_0_" + str(c) + "_breakdown." + fig_fmt), format=fig_fmt, bbox_inches="tight")
    info("Saved figure to %s", os.path.join(fig_dir, "cluster_0_" + str(c) + "_breakdown.") + fig_fmt)

    output_file(os.path.join(fig_dir, "cluster_0_" + str(c) + "_flowchart.html"), title="Cluster 0.0/" + str(c) + " breakdown flowchart (Hemibrain " + hemibrain_version + ")")
    graph, tools, ranges = breakdown_flowchart_graph(FB_node_df[FB_node_df['0.0'] == c][list_of_params],
                                                     max_line_width=80,
                                                     hover_tooltips={"cluster": "@col_value",
                                                                     "param": "@col_name",
                                                                     "size": "@height"})
    plot = figure(title="Cluster 0.0/" + str(c) + " breakdown flowchart",
                  x_range=ranges[0], y_range=ranges[1], plot_width=800, plot_height=800)
    plot.renderers.append(graph)
    plot.add_tools(*tools)
    # show(plot)
    save(plot)
    info("Saved bokeh plot to %s", os.path.join(fig_dir, "cluster_0_" + str(c) + "_flowchart.html"))


# FB_edge_df = pd.read_csv(weighted_edges_csv)
