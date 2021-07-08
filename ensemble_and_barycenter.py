from gerrychain import Graph, Election, updaters, Partition, constraints, MarkovChain
from gerrychain.updaters import cut_edges
from gerrychain.proposals import recom, random
from gerrychain.tree import recursive_tree_part
from gerrychain.accept import always_accept
from functools import partial
import operator
import numpy as np
import pickle
import matplotlib
import littlehelpers
import sys
import math
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ot
import networkx as nx
import geopandas as gpd
import pickle
from scipy.optimize import linear_sum_assignment as LSA
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scipy
import time
from higher_bary import find_barycenter, planar_discrete_barycenter, planar_discrete_distance, k_means, match_to_mean
import argparse, os 
import tracemalloc
np.random.seed(2021)
random.seed(2021)

parser = argparse.ArgumentParser()
parser.add_argument('shapefile', type=str, help='the shapefile to use')
parser.add_argument('output', type=str, help='folder for output')
parser.add_argument('--graph', type=str, help='json file for graph if available')
parser.add_argument('--bypop', action='store_true', help='whether to use population-weighting')
parser.add_argument('--rerunall', action='store_true', help='whether to rerun in full and not use presaved values where available')
parser.add_argument('--M', type=int, help='the number of sampled points to use', default=40)
parser.add_argument('--k', type=int, help='the number of districts per plan', default=4)
parser.add_argument('--steps', type=int, help='the total number of steps in the chain', default=50000)
parser.add_argument('--interval', type=int, help='the interval at which to sample', default=50)
parser.add_argument('--popcol', type=str, help='the population column', default='TOTPOP')
parser.add_argument('--areacol', type=str, help='the area column', default='ALAND10')
parser.add_argument('--epsg', type=str, help='EPSG code for projection', default=5070)
parser.add_argument('--state', type=str, help='two letter code for state (optional)')


args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)
print('INPUT PARAMETERS:')
print(args)

k = args.k
M = args.M
pop_col = args.popcol
pop_tol = 0.02
shapefile = args.shapefile
output = args.output
if args.bypop:
    tag = 'bypop'
else:
    tag = ""
if args.state is None:
    args.state = output[-2:]

epsilon = 1e-7

def plot_discrete_distribution(mu, ax=None, **kwargs):
    if ax==None:
        ax=plt.gca()
    ax.scatter(
        [x[0] for x in mu],
        [x[1] for x in mu],
        **kwargs
    )
    
def plot_distribution_family(mus, cmap=None, ax=None, **kwargs):
    if cmap is None:
        my_cmap = plt.get_cmap('tab20')
    else:
        my_cmap = cmap
    for i, mu in enumerate(mus):
        plot_discrete_distribution(mu, c=np.array([my_cmap(i)]), ax=ax, **kwargs)

def sample_points_from_district(gdf, district_col, district, K, weight_col=None):
    '''
    returns a Mx2 array
    '''
    if 'centroids' not in gdf.columns:
        gdf['centroids'] = gdf.geometry.centroid
    if weight_col == None:
        chosen_units = random.sample(list(gdf[gdf[district_col]==district].index), K)
    if weight_col == 'area':
        total_weight = gdf[gdf[district_col]==district].area.sum()
        chosen_units = random.choices(
            list(gdf[gdf[district_col]==district].index),
            k=M,
            weights=[x/total_weight for x in gdf[gdf[district_col]==district].area]
        )
    else:
        total_weight = gdf[gdf[district_col]==district][weight_col].sum()
        chosen_units = random.choices(
            list(gdf[gdf[district_col]==district].index),
            k=M,
            weights=[x/total_weight for x in gdf[gdf[district_col]==district][weight_col]]
        )
    return np.array([x.coords[0] for x in gdf['centroids'][chosen_units]])


'''
Make the graph from the shapefile
'''
gdf = gpd.read_file(shapefile)
gdf.geometry = gdf.geometry.buffer(0)
if args.epsg is not None:
    gdf = gdf.to_crs('epsg:{}'.format(args.epsg))
gdf['centroids'] = gdf.geometry.centroid
if args.graph is None or args.graph == 'none':
    graph = Graph.from_geodataframe(gdf, ignore_errors=True)
else:
    print('Using pre-made graph')
    graph = Graph.from_json(args.graph)
print("Shapefile loaded and dual graph computed...\nRunning chain...")
if not nx.is_connected(graph):
    print('GRAPH NOT CONNECTED! Using largest connected component.')
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc)

'''
Run the chain or use pre-run
'''
if os.path.exists('{}/parts_k{}.p'.format(output,k)) and not args.rerunall:
    print('Using pre-saved ensemble')
    parts = pickle.load(open('{}/parts_k{}.p'.format(output,k), 'rb'))
else: 
    total_population = sum([graph.nodes[n][pop_col] for n in graph.nodes()])
    pop_target = total_population/k
    myupdaters = {
            "population": updaters.Tally(pop_col, alias="population"),
            "cut_edges": cut_edges,
        }
    myproposal = partial(recom, pop_col=pop_col, pop_target=pop_target, epsilon=pop_tol, node_repeats=2)
    ass = recursive_tree_part(graph, range(k), total_population/k, pop_col, pop_tol)
    initial_partition = Partition(graph, ass, myupdaters)
    myconstraints = [
        constraints.within_percent_of_ideal_population(initial_partition, pop_tol)
    ]
    chain = MarkovChain(
        proposal=myproposal,
        constraints=myconstraints,
        accept=always_accept,
        initial_state=initial_partition,
        total_steps=args.steps
    )
    parts = []
    for index, step in enumerate(chain):
        if index%args.interval == 0:
            print(index, end=' ')
            parts.append(step.assignment)
    pickle.dump(parts, open('{}/parts_k{}{}.p'.format(output, k, tag), 'wb'))
    print('Done generating ensemble...')
for i, p in enumerate(parts):
    gdf['part{}'.format(i)] = [p[x] if x in p else -1 for x in range(len(gdf))]

'''
Sample points
'''
if os.path.exists('{}/samples_k{}_M{}{}.npy'.format(output,k,M,tag)) and not args.rerunall:
    high_res_plans = np.load('{}/samples_k{}_M{}{}.npy'.format(output,k,M,tag))
else: 
    high_res_plans = []
    if args.bypop:
        for i in range(len(parts)):
            print(".", end="")
            high_res_plans.append([
                sample_points_from_district(
                    gdf,
                    'part{}'.format(i),
                    j,
                    args.M,
                    weight_col=pop_col
                ) for j in range(k)
            ])
    else:
        if args.areacol in gdf.columns:
            for i in range(len(parts)):
                print(".", end="")
                high_res_plans.append([
                    sample_points_from_district(
                        gdf,
                        'part{}'.format(i),
                        j,
                        args.M,
                        weight_col=args.areacol
                    ) for j in range(k)
                ])
        else:
            for i in range(len(parts)):
                print(".", end="")
                high_res_plans.append([
                    sample_points_from_district(
                        gdf,
                        'part{}'.format(i),
                        j,
                        args.M,
                        weight_col='area'
                    ) for j in range(k)
                ])
    np.save('{}/samples_k{}_M{}{}.npy'.format(output,k,M,tag), high_res_plans)
print('Done generating sample points')

'''
Compute barycenters
'''
n = int(args.steps/args.interval)
if os.path.exists('{}/barycenters_k{}_M{}_n{}{}.npy'.format(output, k,M,n,tag)) and not args.rerunall:
    B = np.load('{}/barycenters_k{}_M{}_n{}{}.npy'.format(output, k,M,n,tag))[0]
else: 
    print("Computing barycenters...")
    barycenters = []
    indexings = []
    n = len(high_res_plans)
    plans = high_res_plans
    start = time.time()
    B, I, matchings = k_means(
        plans,
        planar_discrete_barycenter,
        planar_discrete_distance,
        q=2, K=1
    )
    barycenters.append(B)
    indexings.append(I)
    end = time.time()
    print("time: {:.2f}".format(end-start))
    np.save('{}/indexings_k{}_M{}_n{}{}.npy'.format(output, k,M,n,tag), indexings)
    np.save('{}/barycenters_k{}_M{}_n{}{}.npy'.format(output, k,M,n,tag), barycenters)

fig, ax = plt.subplots(figsize=(6,6))
mycmap = littlehelpers.gerrybook_cmap
if args.state is not None:
    allstates = gpd.read_file('cb_2020_us_state_500k/cb_2020_us_state_500k.shp')
    wholestate = allstates[allstates['STUSPS']==args.state].to_crs('epsg:{}'.format(args.epsg))
    wholestate.boundary.plot(edgecolor='black', ax=ax, linewidth=1)
plot_distribution_family(B[0], ax=ax, cmap=mycmap, s=5)
ax.axis('off')
plt.savefig('{}/barycenter_M{}_k{}{}.png'.format(output, M, k, tag), dpi=150, bbox_inches='tight')
plt.close()




