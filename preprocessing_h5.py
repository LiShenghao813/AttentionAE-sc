# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:02:17 2022

@author: LSH
"""

import h5py

import anndata as ad
import pandas as pd
import zipfile
def unzip_file(dataname):
    src_dir = './Data/' + dataname + '/'
    zip_name = src_dir + dataname + '.zip'
    dst_dir = './Data/' + dataname
    r = zipfile.is_zipfile(zip_name)
    if r:
        fz = zipfile.ZipFile(zip_name, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')
    
def preprocessing_h5(data_name):
    f = h5py.File('./Data/{}/{}.h5'.format(data_name,data_name),'r') 
    f.keys()
    X = f['X'][:]
    obs = f['Y'][:]
    obs = pd.DataFrame(obs,index=range(len(obs)),columns=['celltype',])
    obs = pd.DataFrame(obs,index=range(obs.shape[0]),columns=['celltype',])
    adata = ad.AnnData(X, obs=obs, dtype='int32')
    adata.write_h5ad("./Data/AnnData/{}.h5ad".format(data_name))
if __name__ == "__main__":
    dataset=['Muraro', 'Quake_10x_Bladder', 'Quake_10x_Limb_Muscle', 'Quake_10x_Spleen', 'Quake_Smart-seq2_Diaphragm', 
             'Quake_Smart-seq2_Limb_Muscle', 'Quake_Smart-seq2_Lung', 'Quake_Smart-seq2_Trachea', 'Romanov']
    
    for i in dataset:
        unzip_file(i)
        preprocessing_h5(i)