import operator
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from gerrychain import Graph, Partition, updaters
from gerrychain.tree import recursive_tree_part
from collections import defaultdict

#color map for Political Geometry book
colors = [[0.8,0.392156862745098,0.325490196078431],
[0.862745098039216,0.713725490196078,0.274509803921569],
[0.36078431372549,0.682352941176471,0.611764705882353],
[0.72156862745098,0.580392156862745,0.713725490196078],
[0.470588235294118,0.419607843137255,0.607843137254902],
[0.313725490196078,0.0588235294117647,0.23921568627451],
[0.803921568627451,0.850980392156863,0.376470588235294],
[0.329411764705882,0.47843137254902,0.250980392156863],
[0.352941176470588,0.694117647058824,0.803921568627451],
[0.682352941176471,0.549019607843137,0.380392156862745],
[0.701960784313725,0.733333333333333,0.823529411764706],
[0.372549019607843,0.376470588235294,0.411764705882353],
[0.701960784313725,0.933333333333333,0.756862745098039],
[0.588235294117647,0.666666666666667,0.290196078431373],
[0.556862745098039,0.16078431372549,0.0980392156862745],
[0.250980392156863,0.376470588235294,0.803921568627451],
[0.92156862745098,0.909803921568627,0.776470588235294],
[0.866666666666667,0.6,0.368627450980392],
[0.368627450980392,0.552941176470588,0.552941176470588],
[0.207843137254902,0.231372549019608,0.498039215686275]]
gerrybook_cmap = matplotlib.colors.ListedColormap(colors)

#functions
def relabel_by_dem_vote_share(part, election):
    '''
    Renumbers districts by DEM vote share, 0-indexed
    '''
    dem_percent = election.percents('Democratic')
    unranked_to_ranked = sorted([(list(part.parts.keys())[x], dem_percent[x])
                                  for x in range(0, len(part))],
                                  key=operator.itemgetter(1))
    unranked_to_ranked_list = [x[0] for x in unranked_to_ranked]
    unranked_to_ranked = {unranked_to_ranked[x][0]:x for x in range(0, len(part))}
    newpart = Partition(part.graph, {x:unranked_to_ranked[part.assignment[x]] for x in part.graph.nodes}, part.updaters)
    return newpart

def plot_districts_and_labels(part, gdf, labels, cmap="tab20c"):
    '''
    Plots districts with labels on them

    :param part: a partition
    :param gdf: a geodataframe matching part
    :param labels: a dictionary matching districts to strings
    '''
    gdf["assignment"] = [part.assignment[x] for x in part.graph.nodes]
    districts = gdf.dissolve(by="assignment")
    centroids = districts.geometry.representative_point()
    districts["centroid"] = centroids
    fig, ax = plt.subplots(figsize=(20,20))
    part.plot(gdf, cmap=cmap, ax=ax)
    districts.boundary.plot(ax=ax, edgecolor='black')
    for idx, row in districts.iterrows():
        ax.annotate(s=str(labels[row.name]), xy=row['centroid'].coords[0],
                 horizontalalignment='center')
    plt.show()
    del gdf["assignment"]

def split_districts(part, factor, pop_col, pop_tol):
    '''
    Takes a districting plan and splits each district into smaller ones
    '''
    #must be 0-indexed!
    ass = {}
    graph = part.graph
    for d in part.parts:
        subgraph = graph.subgraph(part.parts[d])
        subdistricts = recursive_tree_part(
            subgraph,
            range(factor),
            sum(subgraph.nodes[n][pop_col] for n in subgraph.nodes)/factor,
            pop_col,
            pop_tol
        )
        for x in subgraph.nodes:
            ass[x] = d*factor+subdistricts[x]
    return ass

def factor_seed(graph, k, pop_tol, pop_col):
    '''
    Recursively partitions a graph into k districts.

    Returns an assignment.
    '''
    total_population = sum([graph.nodes[n][pop_col] for n in graph.nodes])
    pop_target = total_population/k
    ass = {x:0 for x in graph.nodes}
    #first determine how many times we will split to set custom pop_tol
    num_d = 1
    factors = 0
    while num_d != k:
        for r in range(2, int(k/num_d)+1):
            if int(k/num_d) % r == 0:
                factors += 1
                num_d *= r
    #now do the split, with a custom pop_tol
    custom_pop_tol = (1+pop_tol)**(1/factors)-1
    num_d = 1
    while num_d != k:
        for r in range(2, int(k/num_d)+1):
            if int(k/num_d) % r == 0:
                print("Splitting from {:d} down to {:d}".format(
                        int(num_d),
                        int(r*num_d)
                        )
                     )
                ass = split_districts(
                    Partition(graph, ass),
                    r,
                    pop_col,
                    custom_pop_tol
                )
                num_d *= r
                break
    #find the population deviation
    pops = defaultdict(float)
    for node, a in ass.items():
        pops[a] += graph.nodes[node][pop_col]
    print("Max deviation: {:.3f}".format(
            max(np.abs(x-pop_target)/pop_target for x in pops.values())
        )
    )

    return ass
