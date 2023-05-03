# flybrain-clustering
Clustering analysis of the *Drosophila melanogaster* connectome. See the preprint on biorxiv [here.](https://www.biorxiv.org/content/10.1101/2022.11.23.517722v1)

The notebooks in this package can be used to produce the figures in the paper. The data, in `/hemibrain` and `/larval_MB`, includes the cluster identities determined by maximizing [generalized modularity density](https://github.com/prameshsingh/generalized-modularity-density).

Any questions can be posted as issues on this repo, or directed to [Alex Kunin](https://github.com/sekunder)


# Set up

## Packages
You will need the following python packages:
* [`neuprint`](https://github.com/connectome-neuprint/neuprint-python)
* [`ipyvolume`](https://ipyvolume.readthedocs.io/en/latest/install.html)
* [`bokeh`](https://docs.bokeh.org/en/2.4.3/docs/first_steps.html)
* [`colorcet`](https://colorcet.holoviz.org/)


## Neuprint Auth token
In order to use these notebooks, you will need an authorization token to access NeuPrint.

1. Go to [neuprint.janelia.org](https://neuprint.janelia.org/)
2. Log in with your Google account.
3. Go to your account (menu in the top right of the screen)
4. Copy the auth token to a plain text file in the same directory as the notebooks and name it `flybrain.auth`

## Notebook information

The notebooks in this repo produce the figures specified below
* `larvel-MB-figure.ipynb` - Figure 1, giving an overview of modularity in the [larval mushroom body](https://www.nature.com/articles/nature23455)
* `overview-figure.ipynb` - Figures 2 and 4, giving an overview of modularity in the [Hemibrain](https://www.janelia.org/project-team/flyem/hemibrain) overall and in the fan-shaped body specifically.
* `reduced-graphs.ipynb` - Figure 3, showing reduced networks where nodes represent clusters, and edges represent the weighted connectivity between clusters
* `celltype figures.ipynb` - Figures 5-7, giving an overview of cell type-specific wiring patterns in the hemibrain
* `thresholded-networks.ipynb` - Figure S15 (supplemental), where we show the effects of perturbing the network (dropping low-confidence synapses) on the clustering results we obtain. The perturbed network was obtained using the script `filter_graph.py`