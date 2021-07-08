import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import higher_bary
import random
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import random
import pandas as pd
import seaborn as sns
import importlib
from collections import defaultdict


random.seed(2021)
np.random.seed(2021)

def generate_data(model = 'gaussians',
                  centers=[(0,0), (1,0), (0,1)],
                  variance=0.2,
                  num_points = 20,
                  factor = .5,
                  noise = .05,
                  add_outliers = False,
                  outlier_density = 0.1,
                  outlier_factor = 3):

    """
    Possible models: 'gaussians','circles','moons','aniso','square' 
    """
    
    data = []
    
    if model == 'gaussians':
        
        for i, c in enumerate(centers):
            data.extend(
                np.random.normal(
                    c,
                    [variance]*len(c),
                    size=(int(num_points/3),2)
                )        
            )
                
    elif model == 'circles':
        
        data.extend(datasets.make_circles(n_samples=num_points, factor=factor,
                                      noise= noise)[0])
    elif model == 'moons':
        
        data.extend(datasets.make_moons(n_samples=num_points, noise=noise)[0])
        
    elif model == 'aniso':
        
        random_state = 170
        
        X, y = datasets.make_blobs(n_samples=num_points,random_state = random_state)
        
        transformation = [[0.06, -0.06], [-0.04, 0.08]]
        
        aniso = np.dot(X, transformation)

        data.extend(aniso)
        
    elif model == 'square':
        
        data.extend(np.random.rand(num_points, 2))
        
    if add_outliers:
        
        diameter = np.max(pairwise_distances(np.array(data)))
        mean = np.mean(np.array(data),axis = 0)
        num_outlier_points = int(len(data)*outlier_density)
        
        noise = np.random.normal(mean,[outlier_factor*diameter/2]*2,size=(num_outlier_points,2))
            
        data.extend(noise)
        
        
    data =   np.array(data, dtype=np.float32)
    data = data - [data[:,0].min(), data[:,1].min()]
    data = data/data[:,0].max()
    return np.array(data, dtype=np.float32)


methods = ['kMeans','spectral','agglomerative_ward','agglomerative_single']
methodnames = ['k-means', 'spectral', 'ward agglom.', 'single linkage']
models = ['gaussians','circles','moons','aniso','square' ]
colors = [
        [0.8,0.392156862745098,0.325490196078431],
        [0.862745098039216,0.713725490196078,0.274509803921569],
        [0.36078431372549,0.682352941176471,0.611764705882353],
        [0.72156862745098,0.580392156862745,0.713725490196078],
        [0.470588235294118,0.419607843137255,0.607843137254902]
    ][:len(methods)]

for mod, model in enumerate(models):

    # Main parameters
    N = 10 # number of datasets 
    M = 100 #number of points in each component in the barycenter
    num_points = 5000

    # Outlier parameters
    add_outliers = False
    outlier_density = 0.5
    outlier_factor = 0.5

    # Specific model parameters
    centers=[(0,0), (1,0), (0,1)]
    variance=0.3
    factor = 0.1
    noise = .1

    all_data = generate_data(model = model,
                               centers = centers, 
                               variance = variance,
                               num_points = num_points,
                               factor = factor,
                               noise = noise,
                               add_outliers = add_outliers,
                               outlier_density = outlier_density,
                               outlier_factor = outlier_factor)
    data_sets = []
    for i in range(N):
        choices = np.random.choice(range(len(all_data)), size=int(num_points/N), replace=False)
        data_sets.append([all_data[j] for j in choices])    
    
    #plot all data together
    fig, ax =plt.subplots(figsize=(5,5))
    data = all_data
    ax.scatter(
        [x[0] for x in data],
        [x[1] for x in data],
        c='black',
        s=2
    )
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_xlim(
        -0.1+min(x[0] for x in all_data),
        +0.1+max(x[0] for x in all_data)
    )
    ax.set_yticks([])
    ax.set_ylim(-1.2,2.2)
    ax.set_ylim(
        -0.1+min(x[1] for x in all_data),
        +0.1+max(x[1] for x in all_data)
    )
    if model == 'square':
        ax.set_xticks([0.25,0.5,0.75])
        ax.set_xticklabels([0.25,0.5,0.75])
        ax.tick_params(axis='x', labelsize=20)
    fig.savefig('sample_{}_{}.png'.format(model, add_outliers), bbox_inches = 'tight', dpi=150)

    #Compute barycenters
    K = {'gaussians':3,'circles':2,'moons':2,'aniso':3,'square':3}[model] 

    barycenters = {}
    all_partitions = {}

    importlib.reload(higher_bary)

    for m, method in enumerate(methods):

        print('Starting',method)

        if method == 'kMeans':
            clustered = [KMeans(n_clusters=K).fit(data) for data in data_sets]

            partitions = [[np.array([x for j, x in enumerate(data_sets[i]) if clustered[i].labels_[j] == k]) 
                           for k in range(K)] for i in range(len(data_sets))]
            
            barycenter_seed = []
            for i in range(K):
                choices = np.random.choice(range(len(partitions[0][i])), size=M, replace=True)
                barycenter_seed.append(
                    [partitions[0][i][j] for j in choices]
                )


            B, I = higher_bary.find_barycenter(
                partitions,
                higher_bary.planar_discrete_barycenter,
                higher_bary.planar_discrete_distance_weighted,
                seed_families = barycenter_seed[:K]
            )

            barycenters[method] = B
            all_partitions[method] = partitions

        if method == 'spectral':

            clustered = [SpectralClustering(n_clusters=K,assign_labels = 'discretize').fit(data)
                         for data in data_sets]

            partitions = [[np.array([x for j, x in enumerate(data_sets[i]) if clustered[i].labels_[j] == k]) 
                           for k in range(K)] for i in range(len(data_sets))]
            
            barycenter_seed = []
            for i in range(K):
                choices = np.random.choice(range(len(partitions[0][i])), size=M, replace=True)
                barycenter_seed.append(
                    [partitions[0][i][j] for j in choices]
                )

            B, I = higher_bary.find_barycenter(
                partitions,
                higher_bary.planar_discrete_barycenter,
                higher_bary.planar_discrete_distance_weighted,
                seed_families = barycenter_seed[:K]
            )

            barycenters[method] = B
            all_partitions[method] = partitions

        if method == 'agglomerative_ward':

            clustered = [AgglomerativeClustering(n_clusters=K,linkage = 'ward').fit(data)
                         for data in data_sets]

            partitions = [[np.array([x for j, x in enumerate(data_sets[i]) if clustered[i].labels_[j] == k]) 
                           for k in range(K)] for i in range(len(data_sets))]
            
            barycenter_seed = []
            for i in range(K):
                choices = np.random.choice(range(len(partitions[0][i])), size=M, replace=True)
                barycenter_seed.append(
                    [partitions[0][i][j] for j in choices]
                )

            B, I = higher_bary.find_barycenter(
                partitions,
                higher_bary.planar_discrete_barycenter,
                higher_bary.planar_discrete_distance_weighted,
                seed_families = barycenter_seed[:K]
            )

            barycenters[method] = B
            all_partitions[method] = partitions

        if method == 'agglomerative_single':

            clustered = [AgglomerativeClustering(n_clusters=K,linkage = 'single').fit(data)
                         for data in data_sets]

            partitions = [[np.array([x for j, x in enumerate(data_sets[i]) if clustered[i].labels_[j] == k]) 
                           for k in range(K)] for i in range(len(data_sets))]
            
            barycenter_seed = []
            for i in range(K):
                choices = np.random.choice(range(len(partitions[0][i])), size=M, replace=True)
                barycenter_seed.append(
                    [partitions[0][i][j] for j in choices]
                )

            B, I = higher_bary.find_barycenter(
                partitions,
                higher_bary.planar_discrete_barycenter,
                higher_bary.planar_discrete_distance_weighted,
                seed_families = barycenter_seed[:K]
            )

            barycenters[method] = B
            all_partitions[method] = partitions

        print('...Done!')

    for method in methods:
        fig, ax = plt.subplots(figsize=(5,5))
        B = barycenters[method]
        for b in B:
            plt.scatter(
                [x[0] for x in b],
                [x[1] for x in b],
                s=20
            )  
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_xlim(
            -0.1+min(x[0] for x in all_data),
            +0.1+max(x[0] for x in all_data)
        )
        ax.set_yticks([])
        ax.set_ylim(-1.2,2.2)
        ax.set_ylim(
            -0.1+min(x[1] for x in all_data),
            +0.1+max(x[1] for x in all_data)
        )
        if model == 'square':
            ax.set_xticks([0.25,0.5,0.75])
            ax.tick_params(axis='x', labelsize=20)
        fig.savefig('bary_{}_{}_{}.png'.format(model, add_outliers, method), bbox_inches = 'tight', dpi=150)
        plt.close(fig)
        
    dists = defaultdict(list)
    for method in methods:
        for j in range(N):
            _, d = higher_bary.match_to_mean(all_partitions[method][j], 
                                             barycenters[method], 
                                             higher_bary.planar_discrete_distance_weighted)        
            dists[method].append(d)
            
    width=0.95
    medianprops = dict(color='black')
    boxfig, boxax = plt.subplots(figsize=(5,5))
    boxprops = dict(linestyle='--', linewidth=2, color='black')
    bplot = boxax.boxplot(
        [dists[m] for m in methods],
        positions=[width/2-(m+0.5)*width/len(methods) for m in range(len(methods))],
        patch_artist=True,
        widths = width/len(methods),
        vert = False,
        showfliers=False,  
        medianprops=medianprops
    )   
    
        
    for i, (patch, color) in enumerate(zip(bplot['boxes'], colors)):
        patch.set_facecolor(color)   
    for i, (patch, color) in enumerate(zip(bplot['boxes'], colors)):
        patch.set_edgecolor(color) 
    if model == 'gaussians':
        handles = [mpatches.Patch(color=colors[m], label=method) for m, method in enumerate(methodnames)]
        boxax.legend(handles=handles, loc='upper right',fontsize='xx-large')
    if model == 'square':
        boxax.set_xticks([0.25,0.5,0.75])
        boxax.tick_params(axis='x', labelsize=20)
    else:
        boxax.set_xticks([])
    boxax.set_xlim(0,0.9)
    boxax.set_yticklabels([])
    boxax.set_ylim(-0.5, 0.5)
    boxfig.savefig('distortions_{}.png'.format(model), bbox_inches = 'tight', dpi=150)       
    