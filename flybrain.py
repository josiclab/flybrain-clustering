"""Common core of scripts for analysis of fly brain data"""

import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import DrawingArea, OffsetImage, AnnotationBbox

import time
import datetime

from neuprint import Client
# if missing: https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html
# pip3 install neuprint-python
# conda install -c flyem-forge neuprint-python
from neuprint import fetch_neurons, NeuronCriteria as NC, fetch_adjacencies, merge_neuron_properties

wd = os.getcwd()
os.chdir("/notebooks/ergm")
from ergm import ERGM
from example_stats import u_edge_triangle_density, u_delta_edge_triangle_density, d_connected_triplet_motif_density, \
    d_delta_connected_triplet_motif_density, triplet_over_to_exact
from util import log_msg, ellipse, networkx_graph_to_sparse_array

os.chdir(wd)
del wd


def get_neuprint_client(hemibrain_version="v1.1", auth_token_file="notebooks/nx_graph_utils/flybrain.auth"):
    """Load a NeuPrint `Client` object connected to the hemibrain database"""
    auth_token = next(open(auth_token_file)).strip()

    np_client = Client('neuprint.janelia.org', dataset='hemibrain:' + hemibrain_version, token=auth_token)
    return np_client


def get_FlyBrain_graph(hemibrain_version="v1.1",
                       directory="/notebooks/nx_graph_utils/hemibrain/exported-traced-adjacencies-",
                       nodes_file="traced-neurons.csv",
                       edges_file="traced-total-connections.csv",
                       clusters_dir="/notebooks/nx_graph_utils/hemibrain/und_flybrain_v1.1/"):
    """Load the flybrain graph from the specified files. Returns a Networkx graph with weighted edges and labeled
    nodes. """
    directory = directory + hemibrain_version
    log_msg("Attempting to read edges from", os.path.join(directory, edges_file))

    edge_data = open(os.path.join(directory, edges_file), "r")
    next(edge_data, None)  # skip the first line of the file, which is just a header row

    t_start = time.time()
    FlyBrain = nx.parse_edgelist(edge_data, delimiter=",", create_using=nx.DiGraph, nodetype=int,
                                 data=[("n_synapses", int)])
    t_end = time.time()

    edge_data.close()

    log_msg("Read in {} nodes and {} edges in {} s:".format(nx.number_of_nodes(FlyBrain), nx.number_of_edges(FlyBrain),
                                                            t_end - t_start))

    print(nx.info(FlyBrain))

    node_data = open(os.path.join(directory, nodes_file), 'r')
    # next(node_data,None)

    log_msg("Attempting to read cell type data from {}".format(os.path.join(directory, nodes_file)))
    line_counter = 0
    for line in node_data.readlines()[1:]:
        l = line.split(",")
        ct = l[1].strip()
        inst = l[2].strip()
        if len(ct) == 0:
            ct = "None"
        if len(inst) == 0:
            inst = "None"
        nx.set_node_attributes(FlyBrain, {int(l[0]): {"celltype": ct, "instance": inst}})  # in v1.1 it's switched
    node_data.seek(0)

    n_clusters = {}
    cluster_hist = {}

    for filename in os.listdir(clusters_dir):
        log_msg("Reading file", filename)
        if "unweighted" in filename:
            cluster_name = "unweighted"
        elif "lr" in filename:
            cluster_name = "lr"
        else:
            try:
                cluster_name = (filename.split(".")[0]).split("_")[2]
                cluster_name = cluster_name[1:].replace("p", ".")
            except:
                log_msg("Error reading file", filename, ", skipping file")
                continue
#         lines_read, clust_histogram = read_cluster_ids(FlyBrain, filename, cluster_name)
        cluster_info = open(os.path.join(clusters_dir, filename), "r")
        for nl, cl in zip(node_data.readlines()[1:], cluster_info.readlines()):
            n_id = int(nl.split(",")[0])
            c_id = int(cl)
            nx.set_node_attributes(FlyBrain, {n_id: {cluster_name: c_id}})
        n_clusters[cluster_name] = clust_histogram.shape[0]
        cluster_hist[cluster_name] = clust_histogram
        node_data.seek(0)

    log_msg(n_clusters)
    node_data.close()

    return FlyBrain, n_clusters, cluster_hist


def read_cluster_ids(G, cluster_file, cluster_name,
                     hemibrain_version="v1.1",
                     node_dir="/notebooks/nx_graph_utils/hemibrain/exported-traced-adjacencies-",
                     node_file="traced-neurons.csv",
                     cluster_dir="/notebooks/nx_graph_utils/hemibrain/und_flybrain_v1.1/"):
    node_dir = node_dir + hemibrain_version
    nodes = open(os.path.join(node_dir, node_file), "r")
    clust = open(os.path.join(cluster_dir, cluster_file), "r")
    clust_ids = np.array([int(cl) for cl in clust.readlines()])
    lines_read = 0
    for nl, cl in zip(nodes.readlines()[1:], clust.readlines()):
        n_id = int(nl.split(",")[0])
        c_id = int(cl)
        nx.set_node_attributes(G, {n_id: {cluster_name: c_id}})
        lines_read += 1
    # for nl, c_id in zip(nodes.readlines()[1:], clust_ids):
    #     n_id = int(nl.split(",")[0])
    #     #         c_id = int(cl)
    #     nx.set_node_attributes(G, {n_id: {cluster_name: c_id}})
    #     lines_read += 1
    _, clust_histogram = np.unique(clust_ids, return_counts=True)
    nodes.close()
    clust.close()

    return lines_read, clust_histogram


def load_cluster_info(G=None, clusters_dir="/notebooks/nx_graph_utils/hemibrain/und_flybrain_v1.1/"):
    """Loads cluster info into the provided graph. If no graph is specified, loads the flybrain data with
    `get_FlyBrain_graph`. Returns the graph as well as two dictionaries with information about the clusters. """
    n_clusters = {}
    cluster_hist = {}

    if G is None:
        G = get_FlyBrain_graph()

    for filename in os.listdir(clusters_dir):
        log_msg("Reading file", filename)
        if "unweighted" in filename:
            cluster_name = "unweighted"
        elif "lr" in filename:
            cluster_name = "lr"
        else:
            try:
                cluster_name = (filename.split(".")[0]).split("_")[2]
                cluster_name = cluster_name[1:].replace("p", ".")
            except:
                log_msg("Error reading file", filename, ", skipping file")
                continue
        lines_read, clust_histogram = read_cluster_ids(G, filename, cluster_name)
        n_clusters[cluster_name] = clust_histogram.shape[0]
        cluster_hist[cluster_name] = clust_histogram

    log_msg(n_clusters)

    return G, n_clusters, cluster_hist
