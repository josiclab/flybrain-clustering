"""
This script produces a filtered graph from the Hemibrain connectome.

The nodes will be neurons, edges are (directed) synaptic connections, and edge weight
is the number of synapses between the two neurons. This script will filter out synapses
with confidence below the given threshold(s). Multiple thresholds will be applied in
sorted order; the lowest is used for fetching data from neuprint.janelia.org, then
higher thresholds are applied locally.

Usage:
    python3 filter_graph.py C [C ...] [--version v] [--auth_file file | --auth_token token] [-t | --test]
    python3 filter_graph.py --help
"""

import os
import shutil
import pandas as pd
import numpy as np
import time
import argparse
import logging
from neuprint import Client
from neuprint import fetch_synapse_connections, NeuronCriteria as NC, SynapseCriteria as SC


parser = argparse.ArgumentParser()
parser.add_argument('confidence', type=float, metavar='C', nargs='+',
                    help="Minimum confidence for synapses, pre and post. Passing multiple values will result in multiple output files; the lowest value is passed to a SynapseCriteria to fetch from neuprint; higher values will then be used to filter locally.")
parser.add_argument('--version', default='1.1', help="Hemibrain version (format as version number, with or without leading 'v').")
parser.add_argument('--edge_file', type=str, metavar='FILE', default='traced-total-connections.csv',
                    help="File with original graph edges. 1 / n_batch at a time will be used to limit size of fetch requests.")
parser.add_argument('--n_batch', type=int, metavar='N', default=100, help="Number of pieces to break data into for fetching purposes. Default 100.")
parser.add_argument('--skip_cleanup', action='store_true', help="Skip cleanup step, i.e. don't delete all the synapse files.")
# parser.add_argument('--pre', default=0.0,
#                     help="Minimum confidence for pre-synapses; value will be max of --pre and --confidence")
# parser.add_argument('--post', default=0.0,
#                     help="Minimum confidence for post-synapses; value will be max of --post and --confidence")
parser.add_argument('--auth_file', default='flybrain.auth', help='File with auth token for neuprint.janelia.org')
parser.add_argument('--auth_token', default='', help='Auth token for neuprint.janelia.org; overrides --auth_file')
parser.add_argument('--loglevel', choices=["debug", "info", "warning", "error", "critical"], default="info",
                    help="Level of verbosity. Uses `logging` package. Using 'debug' will result in *very* messy output! Set to 'warning' or 'error' to suppress all but the most critical messages.")
parser.add_argument('-t', '--test', action='store_true', help="Run test, using only a small subset of neurons.")

args = parser.parse_args()

# configure logging
loglevel = args.loglevel.upper()
logging.basicConfig(format="%(asctime)s  %(module)s> %(levelname)s: %(message)s",
                    datefmt="%Y %m %d %H:%M:%S", level=getattr(logging, loglevel))


# Establish connection to neuprint server
hemibrain_version = "v" + args.version.replace('v', '')
logging.info('Hemibrain version: ' + hemibrain_version)

if args.auth_token:
    auth_token = args.auth_token.strip()
else:
    auth_token_file = open(args.auth_file, 'r')
    auth_token = next(auth_token_file).strip()
np_client = Client('neuprint.janelia.org', dataset='hemibrain:' + hemibrain_version, token=auth_token)


confidences = list(sorted(args.confidence))
# pre_confidence = max(args.pre, confidence)
# post_confidence = max(args.post, confidence)

logging.info('Confidence threshold(s): ' + str(confidences))

# set up batch size and looping conditions
edge_file = args.edge_file
try:
    edge_df = pd.read_csv(edge_file)
    logging.info("Read %d edges from %s" % (edge_df.shape[0], edge_file))
except Exception as e:
    logging.error("Error reading edge file %s" % edge_file)
    logging.error(str(e))
    exit(1)

# set up aux directory
aux_dir = 'filter_graph_' + str(min(confidences))
if not os.path.isdir(aux_dir):
    logging.info('Creating auxilliary directory ' + aux_dir)
    os.mkdir(aux_dir)
else:
    logging.info('Found auxilliary directory ' + aux_dir)

n_batch = args.n_batch
batch_size = np.ceil(edge_df.shape[0] / n_batch).astype(int)

if args.test:
    batch_range = [min(n_batch, 17)]
else:
    batch_range = range(n_batch)

syn_criteria = SC(confidence=min(confidences))

missing_batches = []
batch_dfs = []

# first attempt: keep all synapse tables in memory; concatenate with pd.concat
logging.info("Beginning loop over %d batches" % len(batch_range))
for batch in batch_range:
    batch_file = 'batch_' + str(batch) + '.csv'
    if os.path.isfile(os.path.join(aux_dir, batch_file)):
        logging.info("Batch file %d (%s) found, loading from file" % (batch, batch_file))
        try:
            df = pd.read_csv(os.path.join(aux_dir, batch_file))
            batch_dfs.append(df)
        except Exception as e:
            logging.warning("Failed to read batch file %d (%s)" % (batch, batch_file))
            logging.warning(str(e))
            missing_batches.append(batch)
        continue
    try:
        # try to fetch synapses for just the given batch of edges
        logging.info("No file for batch %d found. Fetching from neuprint." % batch)
        sub_df = edge_df.iloc[(batch * batch_size):((batch + 1) * batch_size)]
        pres = sub_df.bodyId_pre.unique()
        posts = sub_df.bodyId_post.unique()
        start = time.time()
        connection_df = fetch_synapse_connections(source_criteria=pres, target_criteria=posts,
                                                  synapse_criteria=syn_criteria)
        end = time.time()
        logging.debug("Fetched %d synapses in %8.2f s" % (connection_df.shape[0], end - start))
        batch_dfs.append(connection_df)
        try:
            connection_df.to_csv(os.path.join(aux_dir, batch_file))
        except Exception as e:
            logging.error("Failed to write synapse batch %d to %s" % (batch, os.path.join(aux_dir, batch_file)))
            logging.error(str(e))
            logging.error("Exiting now. Re-run with same arguments to preserve progress.")
            exit(2)
    except Exception as e:
        # don't save the output, add the batch to the missing batches
        logging.warning("Error during batch %d: %s" % (batch, str(e)))
        missing_batches.append(batch)

if len(missing_batches) > 0:
    logging.warning("Failed to process some batches of edges: " + str(missing_batches))
    logging.warning("Exiting now. Re-run with the same arguments to attempt to fetch missing batches.")
    exit(0)

# at this point, we should have all the synapses, with some possible
# redundancy. So, the process now is: loop through the batches,
# filter by confidence, group by pre and post. Then, concatenate all those
# dfs, then group by pre and post again, this time taking the first
# weight for each (pre,post) pair.

for c in confidences:
    graph_file = 'filtered_connections_confidence_' + str(c) + '.csv'
    logging.info("Filtering and combining data for threshold %f" % c)
    filtered_dfs = [df[(df.confidence_pre >= c) & (df.confidence_post >= c)].groupby(['bodyId_pre', 'bodyId_post']).agg({'roi_pre': 'count'}).reset_index() for df in batch_dfs]
    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    combined_df.groupby(['bodyId_pre', 'bodyId_post']).agg({'roi_pre': 'first'})
    combined_df.rename(columns={'roi_pre': 'weight'}, inplace=True)
    combined_df.reset_index().to_csv(graph_file, index=False)
    logging.info("File saved to %s" % graph_file)
    # counted_connections = connection_df[(connection_df.confidence_pre >= c) & (connection_df.confidence_post >= c)].groupby(['bodyId_pre', 'bodyId_post']).agg({'roi_pre': 'count'})

    meta_info = open('meta_info_confidence_' + str(c) + '.txt', 'w')
    print('Hemibrain Version:', hemibrain_version, file=meta_info)
    print('Confidence Threshold:', c, file=meta_info)
    # print('Source criteria:', source_criteria, file=meta_info)
    # print('Target criteria:', target_criteria, file=meta_info)
    # print('Run time:', end - start, 's', file=meta_info)
    # print('Fetched', select.sum(), 'synapses total', file=meta_info)
    # if i > 0:
    #     print('Fetched', select.sum(), 'synapses total', file=meta_info)
    # else:
    #     print('Fetched', connection_df.shape[0], 'synapses total', file=meta_info)
    print('Filtered graph has', combined_df.shape[0], 'edges', file=meta_info)

if not args.skip_cleanup:
    logging.info("Cleaning up")
    shutil.rmtree(aux_dir)
    # os.rmdir(aux_dir)


# if args.test:
#     source_criteria = NC(rois=['FB'])
#     target_criteria = NC(rois=['FB'])
# else:
#     source_criteria = None
#     target_criteria = None

# syn_criteria = SC(confidence=min(confidences))

# logging.info('Source criteria:  ' + str(source_criteria))
# logging.info('Target criteria:  ' + str(target_criteria))
# logging.info('Synapse criteria: ' + str(syn_criteria))

# logging.info('Fetching synaptic connections...')
# start = time.time()
# connection_df = fetch_synapse_connections(source_criteria=source_criteria, target_criteria=target_criteria,
#                                           synapse_criteria=syn_criteria, batch_size=10)
# end = time.time()
# logging.info('Fetched ' + str(connection_df.shape[0]) + ' synapses in ' + str(end - start) + ' s')

# for i, c in enumerate(sorted(confidences)):
#     logging.info('Constructing graph with confidence threshold ' + str(c))
#     if i > 0:
#         select = (connection_df.confidence_pre >= c) & (connection_df.confidence_post >= c)
#         counted_connections = connection_df[select].groupby(['bodyId_pre', 'bodyId_post']).agg({'roi_pre': 'count'})
#     else:
#         counted_connections = connection_df.groupby(['bodyId_pre', 'bodyId_post']).agg({'roi_pre': 'count'})
#     logging.info('Weighted graph has ' + str(counted_connections.shape[0]) + ' edges')
#     counted_connections.rename(columns={'roi_pre': 'weight'}, inplace=True)

#     counted_connections.reset_index().to_csv('filtered_connections_confidence_' + str(c) + '.csv', index=False)

#     meta_info = open('meta_info_confidence_' + str(c) + '.txt', 'w')
#     print('Hemibrain Version:', hemibrain_version, file=meta_info)
#     print('Confidence Threshold:', c, file=meta_info)
#     print('Source criteria:', source_criteria, file=meta_info)
#     print('Target criteria:', target_criteria, file=meta_info)
#     print('Run time:', end - start, 's', file=meta_info)
#     # print('Fetched', select.sum(), 'synapses total', file=meta_info)
#     if i > 0:
#         print('Fetched', select.sum(), 'synapses total', file=meta_info)
#     else:
#         print('Fetched', connection_df.shape[0], 'synapses total', file=meta_info)
#     print('Filtered graph has', counted_connections.shape[0], 'edges', file=meta_info)
