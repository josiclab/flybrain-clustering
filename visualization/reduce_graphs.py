import pandas as pd
import numpy as np

from functools import reduce


def reduced_graph(node_df, edge_df, col, suffixes=None,
                  count_col='instance',
                  u_col="node1", v_col="node2", weight_col="total_weight",
                  cluster_node_threshold=0, cluster_edge_threshold=0, edge_weight_threshold=0,
                  undirected=True):
    """Use `cols` to group `node_df` and `edge_df`. Column `node_df[col]` should have a labeling of nodes.
    If `suffixes=None`, Attempt to merge `node_df[col]` to `edge_df` along `u_col` and `v_col` to reduce
    the edges. Otherwise, assume that `edge_df[u_col+suffixes[0]]` and `edge_df[v_col+suffixes[1]]` have
    the labels of the nodes.

    Return two dataframes, reduced_nodes and reduced_edges"""
    if suffixes is None:
        suffixes = ["_u", "_v"]
        edge_df = edge_df.merge(node_df[col], left_on=u_col, right_index=True).rename(columns={col: col + suffixes[0]})
        edge_df = edge_df.merge(node_df[col], left_on=v_col, right_index=True).rename(columns={col: col + suffixes[1]})

    # just in case, let's filter out edges that might stay within the right set of clusters
    # but don't involve valid nodes
    edge_df = edge_df[edge_df[u_col].isin(node_df.index) & edge_df[v_col].isin(node_df.index)]

    # group the nodes and throw out small clusters (by number of nodes)
    rn = node_df.groupby(col).agg({count_col: "count"}).rename(columns={count_col: "n_nodes"})
    rn = rn[rn["n_nodes"] >= cluster_node_threshold]

    # throw out the edges below a certain weight or not in the node df, then group the edges, then sort if undirected
    ru_col, rv_col = col + suffixes[0], col + suffixes[1]
    re = edge_df[edge_df[weight_col] >= edge_weight_threshold]
    re = re[(re[ru_col].isin(rn.index)) & (re[rv_col].isin(rn.index))]
    re = re.groupby([ru_col, rv_col]).agg({u_col: "count", weight_col: "sum"}).rename(columns={u_col: "n_edges"}).reset_index()
    if undirected:
        re[[ru_col, rv_col]] = np.sort(re[[ru_col, rv_col]].values, axis=1)
        re = re.groupby([ru_col, rv_col]).agg({"n_edges": "sum", weight_col: "sum"}).reset_index()

    # merge in the cluster sizes and compute possible number of edges
    re = re.merge(pd.DataFrame(node_df.value_counts(col)), left_on=ru_col, right_index=True).rename(columns={0: "n_cluster1"})
    re = re.merge(pd.DataFrame(node_df.value_counts(col)), left_on=rv_col, right_index=True).rename(columns={0: "n_cluster2"})
    re["possible_edges"] = re["n_cluster1"] * re["n_cluster2"]
    if undirected:
        re["possible_edges"] = re["possible_edges"].where(re[ru_col] != re[rv_col],
                                                          (re["possible_edges"] - re["n_cluster1"]) // 2)
    else:
        re["possible_edges"] = re["possible_edges"].where(re[ru_col] != re[rv_col],
                                                          re["possible_edges"] - re["n_cluster1"])

    # merge some cluster info into the reduced node df
    rn = rn.merge(re[re[ru_col] == re[rv_col]][[ru_col, "n_edges", weight_col, "possible_edges"]],
                  left_index=True, right_on=ru_col, how="left").rename(columns={ru_col: col}).set_index(col)
    rn = rn.fillna(0)
    rn = rn[rn["n_edges"] >= cluster_edge_threshold]

    # throw in the edge density and weighted edge density, why not
    rn["edge_density"] = rn["possible_edges"].where(rn["possible_edges"] == 0, rn["n_edges"] / rn["possible_edges"])
    rn["weighted_density"] = rn["possible_edges"].where(rn["possible_edges"] == 0, rn[weight_col] / rn["possible_edges"])
    re["edge_density"] = re["possible_edges"].where(re["possible_edges"] == 0, re["n_edges"] / re["possible_edges"])
    re["weighted_density"] = re["possible_edges"].where(re["possible_edges"] == 0, re[weight_col] / re["possible_edges"])

    return rn, re


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


def roi_adjacency(criteria, df, rois=None, merge_cols=['instance', 'celltype', 'type_group']):
    if rois is None:
        rois = get_rois(criteria, df)
    roi_df = pd.DataFrame(index=df[criteria].index)
    roi_df = roi_df.merge(df[merge_cols + ['roiInfo']], left_index=True, right_index=True, how='left')
    roi_df['roiInfo'] = roi_df['roiInfo'].apply(eval)
    roi_df['total_syns'] = 0
    for roi in rois:
        roi_df[roi + '_tbars'] = roi_df['roiInfo'].apply(lambda d: d.get(roi, {}).get('pre', 0))
        roi_df[roi + '_psds'] = roi_df['roiInfo'].apply(lambda d: d.get(roi, {}).get('post', 0))
        roi_df[roi + '_total'] = roi_df[roi + '_tbars'] + roi_df[roi + '_psds']
        roi_df[roi + '_io_ratio'] = (roi_df[roi + '_tbars'] - roi_df[roi + '_psds']) / roi_df[roi + '_total']
        roi_df['total_syns'] = roi_df['total_syns'] + roi_df[roi + '_total']
    for roi in rois:
        roi_df[roi + '_syn_fraction'] = roi_df[roi + '_total'] / roi_df['total_syns']
    return roi_df


def get_rois(df, criteria, filter_list, succs, collapse=[], roiInfo='roiInfo'):
    """Select a subset of `df` using `criteria`, then get all the rois that the selected set of neurons
    actually touch. Select only those rois in `filter_list`, which is `leaves` by default.

    If `collapse` is not empty, if should be a list of ROIs whose children (rois `r` such that `pred[r]`
    is in `collapse`) should be condensed to the parent node."""
    temp_rois = reduce(np.union1d, [list(d) for d in df.loc[criteria, roiInfo].apply(eval)])
    filter_replace = {}
    for p in collapse:
        for c in succs[p]:
            filter_replace[c] = p
    filter_list = [filter_replace.get(r, r) for r in filter_list]
    filter_list = [s for n, s in enumerate(filter_list) if s not in filter_list[:n]]

    selected_rois = [roi for roi in filter_list if roi in temp_rois]
    return selected_rois


def subgraph(V, E, nodes, u_col="start", v_col="end"):
    """Given a graph represented by dataframes V,E return the subgraph on `nodes`
    `nodes` should be a list or an array or some such, not a whole dataframe."""
    Vp = V.loc[nodes]
    Ep = E[E[u_col].isin(nodes) & E[v_col].isin(nodes)]
    return Vp, Ep
