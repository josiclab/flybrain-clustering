import pandas as pd

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
                  u_col="pre", v_col="post", weight_col="total_weight",
                  reset_type=None, additional_node_columns=[], property_name="property"):
    """
    Given a digraph in the form of a node and edge dataframe, with some sort of
    categorical data in `node_df[column]`, return a node dataframe with
    "cluster codes". What this means is, for the distinct values in
    the column, create columns for each, then populate those columns with the
    total weight, or total count, or both, for pre- and post- neighbors.
    """
    post_codes = one_direction_codes(node_df, edge_df, column, u_col, v_col, weight_col)
    pre_codes = one_direction_codes(node_df, edge_df, column, v_col, u_col, weight_col)
    cluster_codes = post_codes.merge(pre_codes, left_index=True, right_index=True, how="outer").fillna(0)

    if reset_type is not None:
        for c in cluster_codes.columns:
            cluster_codes[c] = cluster_codes[c].astype(reset_type)

    small_df = node_df[[column] + additional_node_columns].copy()
    small_df.columns = pd.MultiIndex.from_product([["node"], [column] + additional_node_columns], names=[None, property_name])
    # small_df.columns = pd.MultiIndex.from_tuples([("node", column)], names=[None, property_name])

    cluster_codes = small_df.merge(cluster_codes, left_index=True, right_index=True)

    return cluster_codes


def one_direction_codes(node_df, edge_df, category_column, u_col="pre", v_col="post", weight_col="total_weight"):
    """
    Given a digraph in the form of a node and edge dataframe, with categorical
    data in `node_df[category_column]`, return a node dataframe with "cluster
    codes".

    Returns: A dataframe the same length as `node_df`
    """
    merged_edge_df = edge_df.merge(node_df[[category_column]], left_on=v_col, right_index=True).rename(columns={category_column: category_column + "_" + v_col})
    node_code_df = merged_edge_df.pivot_table(index=u_col,
                                              columns=category_column + "_" + v_col,
                                              values=[v_col, weight_col],
                                              aggfunc={v_col: "count", weight_col: "sum"},
                                              fill_value=0)
    node_code_df.rename(columns={v_col: v_col + "_count", weight_col: v_col + "_" + weight_col}, inplace=True)
    node_code_df.index.rename(node_df.index.name, inplace=True)

    return node_code_df
