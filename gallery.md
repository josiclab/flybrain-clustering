## Gallery

We use a clustering algorithm with a control parameter which essentially controls the coarseness of the clustering -- lower values of the parameter produce larger clusters, which tend to break apart into smaller sub-clusters as the parameter increases.

Below is an example of this breakdown. The leftmost bar shows about 900 cells grouped together at the coarsest scale (control parameter = 0),
in cluster #7 (the cluster labels are arbitrary). We'll refer to clusters by "parameter/id", so this cluster is "0.0/7".
As the parameter increases (x axis), the clusters break apart. The height of each block is the proportion of the original ~900 cells
that group together at the new parameter value.
The white margins indicate cells from the rest of the network that are part of the new cluster. So, for example,
when the parameter increase from 0 to 0.05, the original cluster of 900 breaks into two large subclusters (plus a few tiny clusters).
The orange cluster (which is 0.05/9) consists of 475 cells, all of which were part of cluster 0.0/7.
On the other hand, the green block represents only 441 out of 559 cells in cluster 0.05/7, hence the white margins indicating that 118 cells, or about 20%
of this cluster, does not belong to the original cluster 0.0/7.
Looking from left to right, most of the bars are fully shaded -- the take away is, the clustering algorithm finds finer community structure as the control parameter increases.

<img src="figures/cluster_0_7_breakdown.png">

The other noteworthy feature of this plot is the last two columns. "Celltype" is the expert-labeled cell type information annotated by neuroscientists.
Notice that almost all of the blocks are completely shaded -- that is, this clustering algorithm (which only has access to weighted connectivity information)
groups together many cell types.

Clicking the figures below will take you to interactive plots made using the `bokeh` package in Python.
The contain the same information as above, in the form of a flowchart, to show how the blocks are interrelated.
(If the links are taking you to raw html instead of interactive plots, view this page [on github pages](https://josiclab.github.io/flybrain-clustering/gallery.html))

The top left figure is the breakdown of cluster 0.0/7, from left to right.
The top right and bottom left figures show clusters 0.05/9 and 0.05/7, respectively.
The bottom right figure shows cells of type MC61 (notice the rightmost columns are one color).

[<img src="figures/cluster_0_7_flowchart.png" width="45%" />](figures/cluster_0_7_breakdown.html)
[<img src="figures/cluster_0.05_9_flowchart.png" width="45%" />](figures/cluster_0.05_9_breakdown.html)

[<img src="figures/cluster_0.05_7_flowchart.png" width="45%" />](figures/cluster_0.05_7_breakdown.html)
[<img src="figures/cluster_celltype_MC61_flowchart.png" width="45%" />](figures/cluster_celltype_MC61_breakdown.html)
