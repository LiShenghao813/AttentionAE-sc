# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:35:23 2023

@author: LSH
"""

import scanpy as sc
import scvi
import anndata as ad
import time
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Quake_10x_Limb_Muscle',
                        help='name of input file(a h5ad file: Contains the raw count matrix "X",)')

    args = parser.parse_args()
    for i in ['Pancreas_human2', 'Pancreas_human3', 'Pancreas_human4', 'Pancreas_mouse']:
        args.name = i
        ASW = []
        ARI = []
        NMI = []
    # device = torch.device("cuda" if args.cuda else "cpu")
    # simplefilter(action='ignore', category=FutureWarning)
        for j in range(5):
            adata = ad.read('./Data/AnnData/{}.h5ad'.format(args.name))
            Y = adata.obs['celltype']
            random.seed(j)
            start = time.time()
            adata.var_names_make_unique()
            
            adata.layers["counts"] = adata.X.copy()  # preserve counts
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.raw = adata  # freeze the state in `.raw`
            #             sc.pp.highly_variable_genes( # old version of scanpy
            #                 adata,
            #                 n_top_genes=2000,
            #                 subset=True,
            #                 flavor="seurat",
            # #                 layer="counts",
            
            #             )
            sc.pp.highly_variable_genes( # scanpy 1.7
                adata,
                n_top_genes=2000,
                subset=True,
                flavor="seurat_v3",
                layer="counts",
            )
            scvi.data.setup_anndata(adata, layer="counts")
            model = scvi.model.SCVI(adata)
            model.train()
            latent = model.get_latent_representation()
            adata.obsm["X_scVI"] = latent
            adata.layers["scvi_normalized"] = model.get_normalized_expression(
                library_size=10e4)
            
            sc.pp.neighbors(adata, use_rep="X_scVI")
            sc.tl.umap(adata, min_dist=0.2)
            sc.tl.leiden(adata, key_added="leiden_scVI")
            
            pred = adata.obs['leiden_scVI'].to_list()
            pred = [int(x) for x in pred]
            
            elapsed = time.time() - start
            asw = silhouette_score(adata.obsm['X_scVI'], pred)
            ari = adjusted_rand_score(Y, pred)
            nmi = np.around(normalized_mutual_info_score(Y, pred), 5)
            ASW.append(asw)
            ARI.append(ari)
            NMI.append(nmi)
        if i == 'Pancreas_human2':
            df_ari = pd.DataFrame(ARI,index=range(5), columns=[args.name,])
            df_asw = pd.DataFrame(ASW,index=range(5), columns=[args.name,])
            df_nmi = pd.DataFrame(NMI,index=range(5), columns=[args.name,])
            
        else:
            df_ari['%s'%(args.name)] = ARI
            df_asw['%s'%(args.name)] = ASW
            df_nmi['%s'%(args.name)] = NMI
        
        df_ari.to_csv('./results/scvi_ARI2.csv')
        df_asw.to_csv('./results/scvi_ASW2.csv')
        df_nmi.to_csv('./results/scvi_NMI2.csv') 
