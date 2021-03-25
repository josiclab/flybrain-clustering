import pandas as pd
import numpy as np

from functools import reduce

def reduced_graph(node_df, edge_df, columns,
                  u_col="node1", v_col="node2", weight_col="total_weight",
                  col_suffixes=("_u", "_v"),
                  cluster_node_threshold=0, cluster_edge_threshold=0,
                  edge_weight_threshold=0):
    """
    Given a digraph in the form of a node and edge dataframe, return a new node
    and edge dataframe, grouping the nodes by `columns`. The new vertices are
    the unique tuples `t` in `node_df[columns]`, with edge `t0,t1` included if
    any edge `u,v` in the original graph has `u` with values `t0` and `v` with
    values `t1`.

    Returns three dataframes: Nodes, Edges, Self-Loops. The Edges and Self-Loops
    dataframes can be stacked, they have the same columns.
    """
    # get the unique values in the column(s) and how many nodes there are
    if type(columns) is str:
        columns = [columns]
    reduced_nodes_df = pd.DataFrame(node_df[columns].value_counts()).rename(columns={0: "n_nodes"})

    # exclude groups with too few nodes
    reduced_nodes_df = reduced_nodes_df[reduced_nodes_df["n_nodes"] >= cluster_node_threshold]

    # exclude edges with low weight
    heavy_edge_df = edge_df[edge_df[weight_col] >= edge_weight_threshold]

    # get the relevant node information into the edge dataframe
    temp_df = pd.merge(heavy_edge_df, node_df[columns], left_on=u_col, right_index=True)
    temp_df = pd.merge(temp_df, node_df[columns], left_on=v_col, right_index=True, suffixes=col_suffixes)

    # exclude invalid edges (source or destination was dropped from reduced nodes)
    valid_edge_conditions = [temp_df[c + uv].isin(reduced_nodes_df.index.get_level_values(i)) for i, c in enumerate(columns) for uv in col_suffixes]
    valid_edge_condition = reduce(lambda x, y: x & y, valid_edge_conditions)
    temp_df = temp_df[valid_edge_condition]

    # at this point, temp_df lists (almost) all the edges in the original graph.
    # we want to aggregate by the node labels we merged in, to get the reduced edges
    all_reduced_edges_df = temp_df.groupby([c + s for s in col_suffixes for c in columns]).agg({u_col: "count", weight_col: "sum"}).rename(columns={u_col: "n_edges"}).reset_index()

    self_loop_conditions = [all_reduced_edges_df[c + col_suffixes[0]] == all_reduced_edges_df[c + col_suffixes[1]] for c in columns]
    self_loop_condition = reduce(lambda x, y: x & y, self_loop_conditions)
    reduced_loops_df = all_reduced_edges_df[self_loop_condition]
    reduced_edges_df = all_reduced_edges_df[~self_loop_condition]

    # aggregate the self-loop edges and add some of that info to the reduced_nodes_df
    simplified_loops_df = reduced_loops_df.rename(columns={c + col_suffixes[0]: c for c in columns}).drop(columns=[c + col_suffixes[1] for c in columns])
    reduced_nodes_df = reduced_nodes_df.merge(simplified_loops_df, left_index=True, right_on=columns).set_index(columns)
    
    return reduced_nodes_df, reduced_loops_df, reduced_edges_df


# def reduced_graph_dfs(param, node_df=FB_node_df, edge_df=uFB_edge_df,
#                       cluster_size_threshold=0, edge_weight_threshold=0, directed=False,
#                       origin_id="id", origin_u="node1", origin_v="node2", origin_weight_name="total_weight",
#                       cluster_name="cluster", node_count_name="n_nodes",
#                       edge_count_name="n_edges", edge_weight_name="total_weight", edge_density_name="edge_density"):
#     """Construct the node and edge dataframes for a given parameter.

#     The nodes are the clusters. The node df has the cluster, # nodes in the cluster, # edges within cluster, and the total weight of the edges within the cluster.
#     Nodes are only included if they have at least `cluster_size_threshold` nodes.

#     The edges are the combined edges between clusters. If there is any edge from a node in cluster i to a node in cluster j, edge i,j is added to the df.
#     The weight of the edge is the total weight of all edges included this way.

#     Edges are only counted (in both the node and edge df) if they have weight at least `edge_weight_threshold`"""
#     # first we tally the size of each cluster
#     clusters_node_df = pd.DataFrame(node_df[param].value_counts())  # dataframe with 1 column; cluster id is df index
#     clusters_node_df = clusters_node_df.rename(columns={param: node_count_name})
#     clusters_node_df = clusters_node_df[clusters_node_df[node_count_name] >= cluster_size_threshold]
#     # now we've filtered out clusters that are too small

#     temp_df = pd.merge(node_df[param], edge_df, left_index=True, right_on=origin_u).rename(columns={param: "cluster1"})
#     temp_df = pd.merge(node_df[param], temp_df, left_index=True, right_on=origin_v).rename(columns={param: "cluster2"})
#     if not directed:
#         temp_df[["cluster1", "cluster2"]] = np.sort(temp_df[["cluster1", "cluster2"]])
#     temp_df = temp_df[temp_df[origin_weight_name] >= edge_weight_threshold]  # filter out weak connections
#     # this is all of the edges, now with clusters (note the cluster ids have become dissociated from the nodes)

#     # information about edges within clusters get added to the nodes df
#     within_cluster_edges = temp_df[(temp_df["cluster1"] == temp_df["cluster2"]) & (temp_df[origin_weight_name] >= edge_weight_threshold)]
#     within_cluster_edges = within_cluster_edges.groupby("cluster1")
#     within_cluster_edges = within_cluster_edges.agg({origin_u: "count", origin_weight_name: "sum"})
#     within_cluster_edges = within_cluster_edges.rename(columns={origin_u: edge_count_name, origin_weight_name: edge_weight_name})

#     clusters_node_df = pd.merge(clusters_node_df, within_cluster_edges, left_index=True, right_index=True, how="left")
#     clusters_node_df[edge_density_name] = clusters_node_df[edge_count_name] / (clusters_node_df[node_count_name] * (clusters_node_df[node_count_name] - 1))
#     if not directed:
#         clusters_node_df[edge_density_name] = clusters_node_df[edge_density_name] * 2

#     # information about inter-cluster edges get saved in a separate table
#     inter_cluster_edges = temp_df[temp_df["cluster1"] != temp_df["cluster2"]]
#     clusters_edge_df = inter_cluster_edges.groupby(["cluster1", "cluster2"])
#     clusters_edge_df = clusters_edge_df.agg({origin_weight_name: "sum", "cluster2": "count"})
#     clusters_edge_df = clusters_edge_df.rename(columns={"cluster2": edge_count_name, origin_weight_name: edge_weight_name})

#     return clusters_node_df, clusters_edge_df
