import numpy as np
import pandas as pd

# Clusters
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans, Birch, AgglomerativeClustering, estimate_bandwidth

from sklearn.neighbors import kneighbors_graph
# plot
import matplotlib.pyplot as plt
import seaborn as sns

# Accidentes que tengan que ver con curvas con visibilidad restringida

def define_case_3():
    dgt_data = pd.read_csv("../data/accidentes_2013.csv")
    dgt_data.columns = [col.lower() for col in dgt_data]

    case_3_subset = dgt_data[dgt_data['trazado_no_intersec'].str.contains('CURVA')]
    condition_visibility_no_restriction = case_3_subset['visibilidad_restringida'].str.contains('SIN RESTRICCIÓN')
    case_3_subset = case_3_subset[~condition_visibility_no_restriction]

    return case_3_subset

def define_case_4():
    dgt_data = pd.read_csv("../data/accidentes_2013.csv")
    dgt_data.columns = [col.lower() for col in dgt_data]

    case_4_subset = dgt_data[dgt_data['trazado_no_intersec'].str.contains('CURVA')]
    condition_visibility_no_restriction = case_4_subset['visibilidad_restringida'].str.contains('SIN RESTRICCIÓN')
    case_4_subset = case_4_subset[condition_visibility_no_restriction]

    return case_4_subset

def definition_clusters_case_3(data_norm):

    k_means = KMeans(init='k-means++', n_clusters=5, n_init=100, random_state=3425)
    # estimate bandwidth for mean shift
    bandwidth = estimate_bandwidth(data_norm, quantile=0.3)
    ms = MeanShift(bandwidth=bandwidth)
    two_means = MiniBatchKMeans(n_clusters=5,  init='k-means++', random_state=3425)
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(data_norm, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    ward = AgglomerativeClustering(n_clusters=5, linkage='ward')
    brc = Birch(n_clusters = 5, threshold=0.1)

    clustering_algorithms = (
        ('K-Means', k_means),
        ('MiniBatchKMeans',two_means),
        ('MeanShift', ms),
        ('Agglomerative', ward),
        ('Birch', brc)
    )
    
    return clustering_algorithms

# MeanShift
def configuraciones_meanshift(normalized_set): 

    # estimate bandwidth for mean shift
    bandwidth1 = estimate_bandwidth(normalized_set, quantile=0.3)
    bandwidth2 = estimate_bandwidth(normalized_set, quantile=0.4)
    bandwidth3 = estimate_bandwidth(normalized_set, quantile=0.5)


    ms1 = MeanShift(bandwidth=bandwidth1)
    ms2 = MeanShift(bandwidth=bandwidth2)
    ms3 = MeanShift(bandwidth=bandwidth3)

    clustering_algorithms = (
        ('MeanShift-1', ms1),
        ('MeanShift-2', ms2),
        ('MeanShift-3', ms3)
    )
    
    return clustering_algorithms    

# Agglomerative
def configuraciones_agglomerative_connectivity(normalized_set):
    
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(normalized_set, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    
    ward_10 = AgglomerativeClustering(n_clusters=10, linkage='ward')
    ward_10_connectivity = AgglomerativeClustering(n_clusters=10, linkage='ward', connectivity = connectivity)
    ward_15 = AgglomerativeClustering(n_clusters=15, linkage='ward')
    ward_15_connectivity = AgglomerativeClustering(n_clusters=15, linkage='ward', connectivity = connectivity)
    ward_20 = AgglomerativeClustering(n_clusters=20, linkage='ward')
    ward_20_connectivity = AgglomerativeClustering(n_clusters=20, linkage='ward', connectivity = connectivity)

    clustering_algorithms = (
        ('Ward-10', ward_10),
        ('Ward-10-con', ward_10_connectivity),
        ('Ward-20', ward_20),
        ('Ward-20-con', ward_20_connectivity)
    )
    
    return clustering_algorithms  

# Kmeans   
def configuraciones_kmeans(subset):    

    kmeans_10 = KMeans(n_clusters=10,  init='k-means++',n_init = 100, random_state=3425)
    kmeans_20 = KMeans(n_clusters=20,  init='k-means++',n_init = 100, random_state=3425)
    kmeans_30 = KMeans(n_clusters=30,  init='k-means++',n_init = 100, random_state=3425)
    kmeans_40 = KMeans(n_clusters=40,  init='k-means++',n_init = 100, random_state=3425)
    kmeans_50 = KMeans(n_clusters=50,  init='k-means++',n_init = 100, random_state=3425)
    kmeans_60 = KMeans(n_clusters=60,  init='k-means++',n_init = 100, random_state=3425)

    clustering_algorithms = (
        ('Kmeans-10', kmeans_10),
        ('Kmeans-20', kmeans_20),
        ('Kmeans-30', kmeans_30),
        ('Kmeans-40', kmeans_40),
        ('Kmeans-50', kmeans_50),
        ('Kmeans-60', kmeans_60)
    )
    
    return clustering_algorithms    

# MeanShift
def configuraciones_meanshift(normalized_set): 

    # estimate bandwidth for mean shift
    bandwidth1 = estimate_bandwidth(normalized_set, quantile=0.2)
    bandwidth2 = estimate_bandwidth(normalized_set, quantile=0.3)
    bandwidth3 = estimate_bandwidth(normalized_set, quantile=0.4)
    bandwidth4 = estimate_bandwidth(normalized_set, quantile=0.5)

    ms1 = MeanShift(bandwidth=bandwidth1)
    ms2 = MeanShift(bandwidth=bandwidth2)
    ms3 = MeanShift(bandwidth=bandwidth3)
    ms4 = MeanShift(bandwidth=bandwidth4)

    clustering_algorithms = (
        ('MeanShift-02', ms1),
        ('MeanShift-03', ms2),
        ('MeanShift-04', ms3),
        ('MeanShift-05', ms4)
    )
    
    return clustering_algorithms    
        
        
# Birch 
def configuraciones_birch_clusters(threshold=0.01):
    brc_10 = Birch(n_clusters = 10, threshold=threshold)
    brc_20 = Birch(n_clusters = 20, threshold=threshold)
    brc_30 = Birch(n_clusters = 30, threshold=threshold)
    brc_40 = Birch(n_clusters = 40, threshold=threshold)
    brc_50 = Birch(n_clusters = 50, threshold=threshold)
    brc_60 = Birch(n_clusters = 60, threshold=threshold)

    clustering_algorithms = (
        ('Birch-10', brc_10),
        ('Birch-20', brc_20),
        ('Birch-30', brc_30),
        ('Birch-40', brc_40),
        ('Birch-50', brc_50),
        ('Birch-60', brc_60)
    )
    
    return clustering_algorithms    

def configuraciones_birch_threshold(n_clusters):
    brc_01 = Birch(n_clusters = n_clusters, threshold=0.01)
    brc_05 = Birch(n_clusters = n_clusters, threshold=0.05)
    brc_07 = Birch(n_clusters = n_clusters, threshold=0.07)

    #Los añadimos a una lista
    clustering_algorithms = (
        ('Birch-01', brc_01),
        ('Birch-05', brc_05),
        ('Birch-07', brc_07),
    )
    
    return clustering_algorithms    
        
        