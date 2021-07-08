# gerrycenters: Exploring redistricting ensembles with barycenters

###Basic code for generating and analyzing ensembles using geometric averages (barycenters). 
- `ensemble_and_barycenter.py`: generates an ensemble of redistricting plans, samples points from the districts in each plan and computes the barycenter.
- `make_boxplots.ipynb`: a jupyter notebook which takes the output of the script above and plots the vote shares organized by matching to the barycenter.
- `IA_counties` and `IA_results`: folders containing example data for Iowa.
- `higher_bary.py`: a Python library which implements the barycenter-finding algorithm in a general setting.
- `littlehelpers.py`: some auxillary code for plotting districts

###Code for the clustering experiment in the paper by Needham and Weighill.
- `clustering_experiment.py`: clusters subsampled synthetic datasets and computes the barycenter
