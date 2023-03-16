# flybrain-clustering
Clustering analysis of the *Drosophila melanogaster* connectome. See the preprint on biorxiv [here.](https://www.biorxiv.org/content/10.1101/2022.11.23.517722v1)

The notebooks in this package can be used to reproduce the figures in the paper. The data, in `/hemibrain` and `/larval_MB`, includes the cluster identities determined by maximizing [generalized modularity density](https://github.com/prameshsingh/generalized-modularity-density).

Any questions can be posted as issues on this repo, or directed to [Alex Kunin](https://github.com/sekunder)


# Set up

## Packages
You will need the following packages:
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

