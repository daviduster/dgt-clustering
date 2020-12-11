
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, estimate_bandwidth, MeanShift
from scipy.cluster.hierarchy import dendrogram

from scipy.spatial.distance import cdist 

import time
import warnings

# plot
import matplotlib.pyplot as plt
import seaborn as sns

from pract2_utils import measures_silhoutte_calinski, denorm, numero_clusters

def define_case_1():
    dgt_data = pd.read_csv("../data/accidentes_2013.csv")
    dgt_data.columns = [col.lower() for col in dgt_data]

    case_1_subset = dgt_data[dgt_data['provincia'].str.contains('Madrid')]
    condition_week = (case_1_subset['diasemana'] >= 1) & (case_1_subset['diasemana'] <= 5)
    case_1_subset = case_1_subset[condition_week]
    condition_morning = (case_1_subset['hora'] >= 8) & (case_1_subset['hora'] <= 10)
    condition_afternoon = (case_1_subset['hora'] >= 13) & (case_1_subset['hora'] <= 16)
    case_1_subset = case_1_subset[(condition_morning | condition_afternoon)]
    
    return case_1_subset

def define_case_2():
    dgt_data = pd.read_csv("../data/accidentes_2013.csv")
    dgt_data.columns = [col.lower() for col in dgt_data]

    case_1_subset = dgt_data[dgt_data['provincia'].str.contains('Madrid')]
    condition_night = (case_1_subset['hora'] >= 19) & (case_1_subset['hora'] <= 23.59)
    condition_early_morning = (case_1_subset['hora'] >= 0) & (case_1_subset['hora'] <= 6)
    case_1_subset = case_1_subset[(condition_night | condition_early_morning)]
    
    return case_1_subset


    
def definition_clusters(data_norm):
    
    k_means = KMeans(init='k-means++', n_clusters=5, n_init=100, random_state=3425)
    two_means = MiniBatchKMeans(n_clusters=5,  init='k-means++', random_state=3425)
    # estimate bandwidth for mean shift
    bandwidth = estimate_bandwidth(data_norm, quantile=0.3)
    ms = MeanShift(bandwidth=bandwidth)
    ward = AgglomerativeClustering(n_clusters=100, linkage='ward')
    average = AgglomerativeClustering(n_clusters=100, linkage='average')

    clustering_algorithms = (
        ('K-Means', k_means),
        ('MiniBatchMeans',two_means),
        ('MeanShift', ms), 
        ('AgglomerativeWard', ward),
        ('AgglomerativeAverage', average)
    )
    
    return clustering_algorithms

def kmeans_parameters(data, a=1, b=10, n=1):
    distortions, inertia, times = [], [], []
    
    for k in range(a,b,n):
        t0=time.time()
        results = KMeans(n_clusters=k, random_state=3425).fit(data)
        results.fit(data)
        t1=time.time()
        distortions.append(sum(np.min(cdist(data, results.cluster_centers_,'euclidean'),axis=1)) / data.shape[0])
        inertia.append(results.inertia_)
        times.append(t1-t0)

    return distortions, inertia, times

def plot_time(distortions, times, xlabel, ylabel, name_save, a, b, n):
    fig, ax1 = plt.subplots()
    ran = range(a, b, n)
    ax1.set_xlabel('n_clusters', color='b')
    ax1.set_ylabel(xlabel)
    ax1.plot(ran, distortions, 'bx-')

    ax2 = ax1.twinx() 
    ax2.set_ylabel(ylabel, color='r')  
    ax2.plot(ran, times, 'r--')
    plt.savefig(name_save)
    plt.show()

def measures_kmeans_range(data, a=3, b=10, n=1):
    silhouette_scores, calinski_scores  = [], []
    for k in range(a,b,n):
        results = KMeans(n_clusters=k, random_state=3425).fit(data)
        silhouette, calinski = measures_silhoutte_calinski(data, results.labels_)
        silhouette_scores.append(silhouette)
        calinski_scores.append(calinski)

    return silhouette_scores, calinski_scores

def visualize_centroids_kmeans(n_clusters, data_norm, name_save, columns):
    """
    Visualiza los centroides.

    Parametros:

    - centers: centroides.
    - data: listado de atributos.
    - columns: nombres de los atributos.
    """
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100, random_state=3425).fit(data_norm)

    centroids = k_means.cluster_centers_

    sns.set()
    df_centers = pd.DataFrame(centroids,columns=columns)
    centers_desnormal=denorm(centroids, data_norm)
    hm = sns.heatmap(df_centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    hm.set_xticklabels(hm.get_xticklabels(), rotation = 45, fontsize = 8)
    # estas tres lineas las he añadido para evitar que se corten la linea superior e inferior del heatmap
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom, top)
    #hm.set_ylim(bottom + 1, top - 1) for 4 clusters
    hm.figure.tight_layout()
    plt.savefig(name_save)
    return hm

def delete_outliers(prediction, subset, min_size=2):
    X = subset
    k = numero_clusters(prediction)
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(prediction[1], index=X.index, columns=['cluster'])
    #y se añade como columna a X
    X_cluster = pd.concat([X, clusters], axis=1)
    
    #Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
    X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
    k_filtrado = len(set(X_filtrado['cluster']))
    print('''De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos.
          Del total de {:.0f} elementos, se seleccionan {:.0f}'''.format(k,k_filtrado,min_size,len(X),len(X_filtrado)))
    X_filtrado = X_filtrado.drop('cluster', 1)
    
    #X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')
    
    return X_filtrado

def configuraciones_agglomerative():
    ward_5 = AgglomerativeClustering(n_clusters=5, linkage='ward')
    average_5 = AgglomerativeClustering(n_clusters=5, linkage='average')
    
    ward_10 = AgglomerativeClustering(n_clusters=10, linkage='ward')
    average_10 = AgglomerativeClustering(n_clusters=10, linkage='average')
                                            
    ward_15 = AgglomerativeClustering(n_clusters=15, linkage='ward')
    average_15 = AgglomerativeClustering(n_clusters=15, linkage='average') 

    ward_20 = AgglomerativeClustering(n_clusters=20, linkage='ward')
    average_20 = AgglomerativeClustering(n_clusters=20, linkage='average') 

    ward_25 = AgglomerativeClustering(n_clusters=25, linkage='ward')
    average_25 = AgglomerativeClustering(n_clusters=25, linkage='average')

    ward_30 = AgglomerativeClustering(n_clusters=30, linkage='ward')
    average_30 = AgglomerativeClustering(n_clusters=30, linkage='average')
    
    clustering_algorithms = (
        ('Ward-5', ward_5),
        ('Average-5', average_5),
        ('Ward-10', ward_10),
        ('Average-10', average_10),
        ('Ward-15', ward_15),
        ('Average-15', average_15),
        ('Ward-20', ward_20),
        ('Average-20', average_20),
        ('Ward-25', ward_25),
        ('Average-25', average_25),
        ('Ward-30', ward_30),
        ('Average-30', average_30)

    )
    
    return clustering_algorithms

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)