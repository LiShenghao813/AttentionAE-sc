# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:47:20 2022

@author: LSH
"""

import anndata as ad
import scanpy
import pandas as pd
import numpy as np
from collections import Counter
import torch
import random

#constructing the cell-cell graph
def adata_knn(adata, method, n_neighbors, metric):
    if method == 'umap':
        scanpy.pp.neighbors(adata, method = method, n_neighbors= n_neighbors, metric=metric, n_pcs=0)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']
    elif method == 'gauss':
        scanpy.pp.neighbors(adata, knn=False, method = 'gauss', metric=metric, n_pcs=0)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']
    return adj, r_adj

# To load gene expression data file into the (pre-)train function.
def load_data(dataPath, args, knn_method='gauss', metric='cosine', dropout=0):
    adata = ad.read(dataPath + '.h5ad')    
    scanpy.pp.filter_cells(adata, min_genes=1)
    scanpy.pp.filter_genes(adata, min_cells=1)
    adata.raw = adata
    # print(adata)
    adata.X = adata.X.astype(np.float32)
    scanpy.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    scanpy.pp.log1p(adata)
    scanpy.pp.highly_variable_genes(adata, n_top_genes=2500)
    adata.raw.var['highly_variable'] = adata.var['highly_variable']
    adata = adata[:, adata.var['highly_variable']]
    dataMat = adata.X
    rawData = adata.raw[:, adata.raw.var['highly_variable']].X
    celltype = adata.obs['celltype']
    if dropout !=0:
        dataMat, rawData = random_mask(dataMat, rawData, dropout)    
        adata.X = dataMat
    # Construct graph

    adj, r_adj = adata_knn(adata, method = knn_method, n_neighbors = 0, metric=metric)
    
    return adata, rawData, dataMat, celltype, adj, r_adj

# using Leiden algorithm to initialize the clustering centers.
def use_Leiden(features, n_neighbors=20, resolution=1):
    #from https://github.com/eleozzr/desc/blob/master/desc/models/network.py line 241
    adata0=scanpy.AnnData(features)
    if n_neighbors==0:
        scanpy.pp.neighbors(adata0, knn=False, method = 'gauss', metric='cosine', n_pcs=0)
    else:
        scanpy.pp.neighbors(adata0, n_neighbors=n_neighbors, use_rep="X")
    scanpy.tl.leiden(adata0, resolution=resolution)
    Y_pred_init=adata0.obs['leiden']
    init_pred=np.asarray(Y_pred_init,dtype=int)
    features=pd.DataFrame(adata0.X,index=np.arange(0,adata0.shape[0]))
    Group=pd.Series(init_pred,index=np.arange(0,adata0.shape[0]),name="Group")
    Mergefeature=pd.concat([features,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
    return cluster_centers, init_pred

# using spectral clustering to initialize the clustering centers.
def use_SpectralClustering(data, adj, args):
    #from https://github.com/Philyzh8/scTAG/blob/38ca65d781a20c3c058ac1d4e58f6d17aaf89908/train.py#L30 line 87
    from sklearn.cluster import SpectralClustering
    Y_pred_init = SpectralClustering(n_clusters=args.n_clusters,affinity="precomputed", assign_labels="discretize",random_state=0).fit_predict(adj)
    init_pred=np.asarray(Y_pred_init,dtype=int)
    features=pd.DataFrame(data,index=np.arange(0,data.shape[0]))
    Group=pd.Series(init_pred,index=np.arange(0,data.shape[0]),name="Group")
    Mergefeature=pd.concat([features,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
    return cluster_centers,init_pred

def random_downsimpling(data, num_cell):
    '''
    data: AnnData type data
    num_cell: The number of sampled cells.
    '''
    p = num_cell/data.shape[0]
    matrix = pd.DataFrame([np.array(range(data.shape[0])),data.obs['celltype']]).transpose()
    matrix.columns = ['index','celltype']
    sort_matrix = matrix.sort_values(['celltype','index'])
    t_sample = []
    groups = Counter(sort_matrix['celltype'])
    # i = 0
    for j in groups.values():
        sample =[]
        sub_sample = random.sample(range(j), int(j*p))
        # i += j
        for k in range(j):
            if k in sub_sample:
                sample.append(1)
            else:
                sample.append(0)
        t_sample += sample
    sort_matrix['sampling'] = np.array(t_sample,dtype=np.bool8)
    final_sort_matrix = sort_matrix.sort_values(['index'])
    sample = []
    for i in range(data.shape[0]):
        if final_sort_matrix['sampling'][i]:
            sample.append(i)
    new_X = data.X[sample,:]
    new_obs = data.obs.iloc[sample,:]
    new_data_raw_X = data.raw.X[sample,:]
    new_data = ad.AnnData(X = new_X, obs = new_obs, var = data.var)  
    new_data.raw = ad.AnnData(X = new_data_raw_X, obs = new_data.obs, var = data.raw.var)  
    return new_data
    

def random_mask(data, raw_data, p):
    '''
    Before training, the gene expression matrix and the corresponding count matrix 
    are performed same masking to get the results of figure5b
    data: gene expression matrix
    raw_data: the corresponding count matrix of data
    p: the dropout rate
    '''
    new_l =[]
    new_l2 =[]
    for i in range(data.shape[0]):
        l =[]
        l2 =[]
        rowdata = data[i,:]
        rowdata2 = raw_data[i,:]
        range_row = range(data.shape[1])
        sample = random.sample(range_row, int(data.shape[1] * (1-p)))
        for j in range(data.shape[1]):
            if j in sample:
                l.append(rowdata[j])
                l2.append(rowdata2[j])
            else:
                l.append(0)
                l2.append(0)
        new_l.append(np.array(l))
        new_l2.append(np.array(l2))
    new_data = np.array(new_l)
    new_rawdata = np.array(new_l2)    
    return new_data, new_rawdata


#getting predicted cell label from allocation matrix P or Q.
def dist_2_label(p):
    _, label = torch.max(p, dim=1)
    return label.data.cpu().numpy()

# def cluster_acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     from scipy.optimize import linear_sum_assignment as linear_assignment
#     ind = linear_assignment(w.max() - w)
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size 