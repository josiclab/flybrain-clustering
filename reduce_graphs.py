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


def cluster_codes(node_df, edge_df, column,
                  u_col="pre", v_col="node2", weight_col="total_weight",
                  col_suffixes=("_u", "_v")):
    """
    Given a digraph in the form of a node and edge dataframe, with some sort of
    categorical data in `node_df[column]`, return a node dataframe with
    "cluster codes". What this means is, for the distinct values in
    the column, create columns for each, then populate those columns with the
    total weight, or total count, or both, for pre- and post- neighbors.
    """
    # first, merge in the node data to the edge dataframe (similar to the reduced graph function)
    merged_edge_df = edge_df.merge(node_df[[column]], left_on=u_col, right_index=True).merge(node_df[[column]], left_on=v_col, right_index=True, suffixes=col_suffixes)

    # first, let's get the post-codes
    edges_by_source = merged_edge_df.groupby([u_col, column+col_suffixes[0], column+col_suffixes[1]]).agg({weight_col:"sum", v_col:"count"}).reset_index()
    post_code_df = edges_by_source.pivot_table(index=[column+col_suffixes[0], u_col],
                                               columns=column+col_suffixes[1],
                                               values=[weight_col, v_col],
                                               aggfunc={weight_col:"sum", v_col:"sum"},
                                               fill_value=0)
    post_code_df = post_code_df.reset_index().rename(columns={v_col: v_col+"_count", 
                                                              weight_col: v_col+"_weight",
                                                              column+u_col: column,
                                                              u_col: node_df.index.name})

    # now get the pre-codes
    edges_by_sink = merged_edge_df.groupby([v_col, column+col_suffixes[0], column+col_suffixes[1]]).agg({weight_col:"sum", u_col:"count"}).reset_index()
    pre_code_df = edges_by_sink.pivot_table(index=[column+col_suffixes[1], v_col],
                                            columns=column+col_suffixes[0],
                                            values=[weight_col, u_col],
                                            aggfunc={weight_col:"sum", v_col:"sum"},
                                            fill_value=0)
    pre_code_df = pre_code_df.reset_index().rename(columns={u_col: u_col + "_count",
                                                            weight_col: u_col + "_weight",
                                                            column+v_col: column,
                                                            v_col: node_df.index.name})

    code_df = pre_code_df.merge(post_code_df, on=[node_df.index.name, column])

    return code_df





