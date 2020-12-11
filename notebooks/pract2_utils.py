from sklearn.metrics import calinski_harabasz_score, silhouette_score
import numpy as np
import pandas as pd
import seaborn 
from math import floor

# Alghoritms
from sklearn.cluster import KMeans

# plot
import matplotlib.pyplot as plt

import time
import warnings


def to_matrix(df, columns=[]):
    """Devuelve los atributos seleccionados como valores"""
    return df[columns].dropna().values

def norm(data):
    """Normaliza una serie de datos"""
    return (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

def denorm(data, df):
    """
    Permite desnormalizar
    """
    return data*(df.max(axis=0)-df.min(axis=0))+df.min(axis=0)

def measures_silhoutte_calinski(data, labels):
    """
    Devuelve el resultado de evaluar los clusters de data asociados con labels.
    
    Parámetros:
    
    - data vector de datos ya normalizados.
    - labels: etiquetas.
    """
    # Hacemos una muestra de sólo el 20% porque son muchos elementos
    muestra_silhoutte = 0.2 if (len(data) > 10000) else 1.0
    silhouette = silhouette_score(data, labels, metric='euclidean', sample_size=int(floor(data.shape[0]*muestra_silhoutte)))
    calinski = calinski_harabasz_score(data, labels)
    return silhouette, calinski

def print_measure(measure, value):
    """
    Muestra el valor con un número fijo de decimales
    """
    print("{}: {:.3f}".format(measure, value))

def pairplot(df, columns, name_save, labels):
    """
    Devuelve una imagen pairplot.

    Parámetros:

    - df: dataframe
    - columns: atributos a considerar
    - labels: etiquetas
    """
    df_plot = df.loc[:,columns]
    df_plot['clusters'] = labels
    hm = seaborn.pairplot(df_plot, hue='clusters', palette='Paired', plot_kws={"s":25})
    #hm = seaborn.pairplot(df_plot, hue='classif', palette='Paired', diag_kws=dict(bins=10), plot_kws={"s":25})
    hm.savefig(name_save)


def visualize_centroids(centers, data, name_save, columns, width=0.5):
    """
    Visualiza los centroides.

    Parametros:

    - centers: centroides.
    - data: listado de atributos.
    - columns: nombres de los atributos.
    """
    df_centers = pd.DataFrame(centers,columns=columns)
    centers_desnormal=denorm(centers, data)
    hm = seaborn.heatmap(df_centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    hm.set_xticklabels(hm.get_xticklabels(), rotation = 45, fontsize = 8)
    # estas tres lineas las he añadido para evitar que se corten la linea superior e inferior del heatmap
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + width, top - width)
    hm.figure.tight_layout()
    plt.savefig(name_save)
    return hm

def get_predictions(clustering_algorithms, data_norm):

    predictions = []
    times = []
        
    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(data_norm)
        
        
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.fit_predict(data_norm)
            
        t1 = time.time()
        times.append(t1-t0)
        predictions.append((name, y_pred))
                
    return predictions, times


def calcule_measures(data_norm, predictions, times):
    meditions = []
    
    for i, pred in enumerate(predictions):
        muestra_silhoutte = 0.2 if (len(data_norm) > 10000) else 1.0
        silhouette = silhouette_score(data_norm, pred[1], metric='euclidean', sample_size=int(floor(data_norm.shape[0]*muestra_silhoutte)))
        calinski = calinski_harabasz_score(data_norm, pred[1])
        meditions.append((pred[0], numero_clusters(pred),calinski, silhouette, times[i]))
        
    return meditions

def latex_table(measures, columns, index_bool = True):
    df = pd.DataFrame(measures, columns = columns)

    return df.to_latex(index=index_bool)

def latex_table_index(measures, columns, index):
    df = pd.DataFrame(measures, columns = columns, index=index)

    return df.to_latex()

def numero_clusters(prediction):
    y = prediction[1]
    y_labels = np.unique(y)
    y_labels = y_labels[y_labels>=0]
    
    return len(y_labels)