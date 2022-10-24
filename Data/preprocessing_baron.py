# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:51:12 2022

@author: LSH
"""
# from collections import Counter
import anndata as ad
import pandas as pd
import numpy as np
import zipfile
def unzip_file(dataname):
    zip_name = './Baron/' + dataname + '.zip'
    dst_dir = './Baron' 
    r = zipfile.is_zipfile(zip_name)
    if r:
        fz = zipfile.ZipFile(zip_name, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')
for i in ['Baron1','Baron2','Baron3','Baron4','Baron mouse1','Baron mouse2']:
    unzip_file(i)
human1 = pd.read_csv('./Baron/GSM2230757_human1_umifm_counts.csv')
human2 = pd.read_csv('./Baron/GSM2230758_human2_umifm_counts.csv')
human3 = pd.read_csv('./Baron/GSM2230759_human3_umifm_counts.csv')
human4 = pd.read_csv('./Baron/GSM2230760_human4_umifm_counts.csv')
mouse1 = pd.read_csv('./Baron/GSM2230761_mouse1_umifm_counts.csv')
mouse2 = pd.read_csv('./Baron/GSM2230762_mouse2_umifm_counts.csv')
human1 = human1.dropna()
human2 = human2.dropna()
human3 = human3.dropna()
human4 = human4.dropna()
mouse = pd.concat([mouse1, mouse2])
mouse = mouse.dropna()

X = mouse.iloc[:,range(3,mouse.shape[1])]
adata2 = ad.AnnData(X)
celltype2 = np.asarray(mouse['assigned_cluster'])
adata2.obs['celltype'] = celltype2


adata1_1 = ad.AnnData(human1.iloc[:,range(3,human1.shape[1])])
celltype1_1 = np.asarray(human1['assigned_cluster'])
adata1_1.obs['celltype'] = celltype1_1

adata1_2 = ad.AnnData(human2.iloc[:,range(3,human2.shape[1])])
celltype1_2 = np.asarray(human2['assigned_cluster'])
adata1_2.obs['celltype'] = celltype1_2

adata1_3 = ad.AnnData(human3.iloc[:,range(3,human3.shape[1])])
celltype1_3 = np.asarray(human3['assigned_cluster'])
adata1_3.obs['celltype'] = celltype1_3

adata1_4 = ad.AnnData(human4.iloc[:,range(3,human4.shape[1])])
celltype1_4 = np.asarray(human4['assigned_cluster'])
adata1_4.obs['celltype'] = celltype1_4


adata1_1.write_h5ad('./AnnData/Pancreas_human1.h5ad')
adata1_2.write_h5ad('./AnnData/Pancreas_human2.h5ad')
adata1_3.write_h5ad('./AnnData/Pancreas_human3.h5ad')
adata1_4.write_h5ad('./AnnData/Pancreas_human4.h5ad')
adata2.write_h5ad('./AnnData/Pancreas_mouse.h5ad')