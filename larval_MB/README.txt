CHANGELOG: 22 Feb 2022: In skeletons.csv, skeleton id 3756659 was improperly
formatted -- each line started with a single " character and no comma. I
changed it to "No name",3756659 on each line. As far as I can tell that was
the only error in the data. -Alex Kunin

EXTRA: The file "connectivity matrix table 1.csv" has the following description in the article: "This file contains the Connectivity matrix of the entire MB network. Neurons in rows are presynaptic to neurons in columns. PN-PN connections are almost all dendro-dendritic and occur within the antennal lobe. (ZIP 20 kb)". The folder "Data for Figures" is from the same source: "This file contains source data for the Main and Extended Data Figures. (ZIP 19905 kb)"

8 Mar 2022: Note that the file synapses.csv includes *all* synapses in the volume, even if they belong to neurons that are not fully traced out. So, there are substantially more unique skeleton IDs (on the order of 10,000) in synapses.csv as compared to unique skeleton IDs (on the order of 400) in skeletons.csv.

But moreover, *every* synapse connecting two "valid" skeletons (i.e. every synapse whose pre- and post-synaptic skeleton IDs are in skeletons.csv) is duplicated, i.e. it appears twice in the file. Thus, if you, say, use pandas to read in the file, select only the "valid" synapses, and then do some grouping to get the adjacency matrix, all of your entries will be twice what appears in connectivity matrix table 1.csv

# Begin original file

The CSV files describe the anatomy and connectivity of all neurons of the left and right mushroom bodies of a first instar, 6-hour-old Drosophila larva. These neurons were reconstructed by the Authors (see below) using the software CATMAID.

CSV files include all neurons that make or receive at least one synapse to a Kenyon cell, except for the few unresolvable fragments (small skeletons that cannot be merged with proper fully reconstructed neurons due to data artifacts).

Coordinates and radii in nanometers.

Each skeleton models the arbor of a single neuron.

All skeleton IDs and skeleton node IDs are unique. Neuron names might not be unique.

By partitioning the CSV file for the skeletons by skeleton ID, and then using the parent-child relationship of the skeleton nodes, the whole arbor of each neuron can be reconstructed. Each skeleton node has an associated x,y,z coordinate and a radius 'r', the latter only set for the soma (potentially also elsewhere unintentionally). Some neurons might lack a soma due to artifacts in the electron microscopy images. The root node of a skeleton lacks a parent, and is generally set at the soma.

Each row in the CSV file for the synapses relates one skeleton node of one skeleton with one skeleton node of another skeleton, with the first being presynaptic and the second being postsynaptic. Note that one presynaptic node of one skeleton might be related to more than one postsynaptic nodes other skeletons, defining a polyadic synapse. This is almost always the case in the data.

Gap junctions are not included, for they were not visible in the electron microscopy (EM) volume.

The serial section transmission electron microscopy volume was imaged at 3.8x3.8x50 nanometers.

The EM volume exactly corresponding to the same coordinate system of these exported skeleton and synapse data is available at the URL below. The URL will load at 100% magnification, conveniently centered at the right mushroom body calyx (the volume is shown in anterior view, therefore "left" is right):

http://openconnecto.me/catmaid/?pid=59&zp=1017&yp=4559&xp=8251&tool=navigator&sid0=93&s0=2

We, the authors, release the data under a creative commons CC-BY-NC 4.0 license.

Date: May 22nd, 2017.

Contributors (authors of) to the EM reconstruction:

Katharina Eichler
Feng Li
Ingrid Andrade
Timo Saumweber
Avinash Khandelwal
Matthew Berck
Ivan Larderet
Javier Valdes Aleman
Volker Hartenstein
Bruno Afonso
Andreas Thum
Marta Zlatic
Albert Cardona
