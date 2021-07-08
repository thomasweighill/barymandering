import numpy as np
try:
    import ot
except ModuleNotFoundError:
    print("Library pot not found, some functions may not work.")
try:
    import cvxpy
except ModuleNotFoundError:
    print("Library cvxpy not found, some functions may not work.")
import networkx as nx
import pickle
from scipy.optimize import linear_sum_assignment as LSA
import scipy
from networkx import incidence_matrix

def k_means(
	families,
	barycenter_function,
	distance_function,
	q=2,
	K=1,
	seeds=None, seed_families=None,
    eps=1e-7, max_iter=1000,
	verbose=True, 
	):
    '''
    General function for k-means clustering of k-families.
    Notation: we use upper case K for the number of means in "K-means".

    Parameters
    ----------
    families : 
        A list of k-families of distributions.
    barycenter_function :
        A method for finding a local barycenter for a set of distributions
    distance_function :
        A method for computing distances between individual distributions
    K:
    	The number of clusters to find
    seeds : list of ints, optional
        Which families to seed the algorithm on
    seed_families : list of k-families, optional
        A list of k-families to use as a seed, only used if seeds is None
    eps : float, optional
        A convergence threshold measured
    max_iter: int, optional
        Number of iterations to try before declaring no convergence

    Returns
    ---------
    A list of K k-families as means of clusters
    Optimal indexings of each k-family by these means
    '''
    if (seeds is None or seeds == [None]) and (seed_families is None or seed_families == [None]):
        means = families[0:K].copy()
    elif seed_families is None:
        means = [families[s] for s in seeds]
    else:
        means = seed_families

    matchings = [0]*len(families)
    for i in range(max_iter):
        #align each family to each mean
        if verbose:
            print("aligning", end=" | ")
        indexings_and_distances = [
        	[
        		match_to_mean(p, mean, distance_function, q=q) for p in families
        	] for j, mean in enumerate(means)  
        ]

        #match families to means
        if verbose:
            print("matching", end=" | ")
        matchings = [
        	np.argmin([x[i][1] for x in indexings_and_distances]) for i, p in enumerate(families) 
        ]
        indexings = [
        	indexings_and_distances[matchings[i]][i][0] for i, p in enumerate(families)
        ]
        if verbose:
            print('D={:e}'.format(sum(
                        indexings_and_distances[matchings[i]][i][1]**q for i, p in enumerate(families)
                    )**(1/q)
                ), end=" | "
            )

        #adjust mean
        if verbose:
            print("recalculating means", end=" | ")
        newmeans = []
        for m, mean in enumerate(means):
	        newmeans.append([
	            barycenter_function(
	                [p[indexings[j][i]] for j, p in enumerate(families) if matchings[j]==m],
	                seed=mean[i]
	            ) for i in range(len(mean))
	        ])

        #check for convergence
        change = max(distance_function(mean[i], newmean[i]) for mean, newmean in zip(means, newmeans) for i in range(len(mean)))
        if verbose:
            print("iter {}: change to barycenter = {:.2f}".format(i, change))
        if change <= eps:
            return newmeans, indexings, matchings
        else:
            means = newmeans.copy()
    print('Did not converge')
    pickle.dump((newmeans, indexings, matchings), open("non_converge_dumps_M{}_K{}.p".format(len(newmeans[0]), K), "wb"))
    return newmeans, indexings, matchings

def find_barycenter(
    families,
    barycenter_function,
    distance_function,
    q=2,
    seed=None, seed_families=None,
    eps=1e-7, max_iter=1000,
    verbose=True):
    '''
    General function for finding approximate barycenters.

    Parameters
    ----------
    families : 
        A list of k-families of distributions.
    barycenter_function :
        A method for finding a local barycenter for a set of distributions
    distance_function :
        A method for computing distances between individual distributions
    q : int, optional
    	The order of the total distance as an l^q sum
    seed : int, optional
        Which family to seed the algorithm on
    seed_families : list of lists of k-families, optional
        A k-tuple of lists of k-families to use as a seed, only used if seeds is None
    eps : float, optional
        A convergence threshold measured
    max_iter: int, optional
        Number of iterations to try before declaring no convergence

    Returns
    ---------
    A k-family which is a local barycenter
    Optimal indexings of each k-family by this mean
    '''
    newmeans, indexings,_ = k_means(
        families,
        barycenter_function,
        distance_function,
        q=2,
        K=1,
        seeds=[seed], seed_families=[seed_families],
        eps=1e-7,
        max_iter=1000,
        verbose=True, 
    )
    return newmeans[0], indexings

def match_to_mean(family, mean, distance_function, q=2):
    '''
    family is a single k-family
    mean is a single k-family
    returns a permutation P such that mean[i] matches to family[P[i]]
    q is the degree to raise each distance to before adding
    '''
    D = np.array([
            [
                distance_function(p, m)**q for p in family
            ] for m in mean
    ])
    row_ind, col_ind = LSA(D)
    return col_ind, sum(D[r,c] for r, c in zip(row_ind, col_ind))**(1/q)

def planar_discrete_barycenter(distributions, seed=None):
    '''
    distributions is a list of Kx2 arrays
    '''
    if seed is None:
        seed = distributions[0]
    K = len(distributions[0])
    weights = [np.ones(len(d),dtype=np.float32)/len(d) for d in distributions]
    for i in range(len(weights)):
        weights[i][0] = 1-sum(weights[i][1:]+[0])
    B = ot.lp.free_support_barycenter(
        distributions,
        weights,
        np.array(seed)
    )
    return B

def planar_discrete_distance(distribution1, distribution2, q=2):
    '''
    distribution1 and distribution2 are each Kx2 arrays
    '''
    if len(distribution1) != len(distribution2):
        raise ValueError('Two distributions do not have equal support size, use planar_discrete_distance_weighted instead.')
    M = scipy.spatial.distance_matrix(distribution1, distribution2)**q
    row_ind, col_ind = LSA(M)
    return (M[row_ind,col_ind].sum()/len(distribution1))**(1/q)

def planar_discrete_distance_weighted(distribution1, distribution2, weights1=None, weights2=None, q=2):
    '''
    distribution1 and distribution2 are ?x2 arrays
    '''
    if weights1 is None:
        weights1 = np.ones(len(distribution1), dtype=np.float32)/len(distribution1)
    if weights2 is None:
        weights2 = np.ones(len(distribution2), dtype=np.float32)/len(distribution2)
    M = scipy.spatial.distance_matrix(distribution1, distribution2)**q
    if len(weights1) > 1:
        weights1[-1] = 1-sum(weights1[:-1])
    if len(weights2) > 1:
        weights2[-1] = 1-sum(weights2[:-1])
    s, log = ot.lp.emd2(weights1, weights2, M.astype(np.float64), log=True)
    if log['warning'] is not None:
        print(log['warning'])
    return s**(1/q)


def graph_W1_barycenter(distributions, edge_incidence=None, graph=None, seed=None):
    '''
    distributions is a list of indicator functions
    '''
    if incidence_matrix is None:
        edge_incidence = incidence_matrix(graph, oriented=True)
    n_edges = edge_incidence.shape[1]
    edge_weights = cvxpy.Variable((len(distributions), n_edges)) #one set of edge flows per district
    bary_indicator = cvxpy.Variable(edge_incidence.shape[0]) #the location of the barycenter which we solve for
    objective = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(edge_weights))) #minimize sum of W1 distances barycenter<->districts
    constraints = []
    for i in range(len(distributions)):
        if isinstance(distributions[i], scipy.sparse.csr.csr_matrix):
            constraints.append(
                (edge_incidence @ edge_weights[i]) == distributions[i].asformat('array')[0] - bary_indicator
            ) #flows source and sink
        elif len(distributions[i].shape) > 1:
            distributions[i] = np.array(distributions[i])[0]
        else: 
            constraints.append(
                (edge_incidence @ edge_weights[i]) == distributions[i] - bary_indicator
            ) #flows source and sink
    constraints.append(cvxpy.sum(bary_indicator) == 1) #barcenter sum to on2
    constraints.append(bary_indicator >= 0) #positive
    prob = cvxpy.Problem(objective, constraints) 
    prob.solve(solver='ECOS')  # solver recommended by Zach for big graphs
    return bary_indicator.value

def graph_W1_distance(distribution1, distribution2, edge_incidence=None, graph=None):
    '''
    distribution1 and distribution 2 are indicator functions
    '''
    if incidence_matrix is None:
        edge_incidence = incidence_matrix(graph, oriented=True)
    n_edges = edge_incidence.shape[1]
    edge_weights = cvxpy.Variable(n_edges)
    diff = (distribution2 - distribution1)
    if isinstance(diff, scipy.sparse.csr.csr_matrix):
        diff = diff.asformat('array')[0]
    elif len(diff.shape) > 1:
        diff = np.array(diff)[0]
    objective = cvxpy.Minimize(cvxpy.sum(cvxpy.abs(edge_weights)))
    conservation = (edge_incidence @ edge_weights) == diff
    prob = cvxpy.Problem(objective, [conservation])
    prob.solve(solver='ECOS')  # solver recommended by Zach for big graphs
    return np.sum(np.abs(edge_weights.value))

def indicators_from_graph_attribute(attribute, graph):
    '''
    turns a categorical variable on a graph into a list of indicators
    '''
    values = set([graph.nodes[n][attribute] for n in graph.nodes])
    indicators = []
    for value in values:
        a = np.array([int(graph.nodes[n][attribute]==value) for n in graph.nodes])
        indicators.append(a/sum(a))
    return indicators

