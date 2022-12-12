# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:34:50 2022

@author: LSH
"""
import torch
from warnings import simplefilter 
import argparse
from sklearn import preprocessing
import random
import numpy as np
import utils
from model import AttentionAE
from train import train, clustering
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001,help='learning rate, default:1e-3')
    parser.add_argument('--n_z', type=int, default=16, 
                        help='the number of dimension of latent vectors for each cell, default:16')
    parser.add_argument('--training_epoch', type=int, default=200,
                        help='epoch of train stage, default:200')
    parser.add_argument('--clustering_epoch', type=int, default=100,
                        help='epoch of clustering stage, default:100')
    parser.add_argument('--name', type=str, default='Muraro',
                        help='name of input file(a h5ad file: Contains the raw count matrix "X",)')
    parser.add_argument('--max_num_cell', type=int, default=4000,
                        help='''a maximum threshold about the number of cells use in the model building, 
                        4,000 is the maximum cells that a GPU owning 11 GB memory can handle. 
                        More cells will bemploy the down-sampling straegy, 
                        which has been shown to be equally effective,
                        but it's recommended to process data with less than 24,000 cells at a time''')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='use GPU, or else use cpu (setting as "False")')
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    simplefilter(action='ignore', category=FutureWarning)
    
    random.seed(1000)
    adata, rawData, dataset, celltype, adj, r_adj = utils.load_data('./Data/AnnData/{}'.format(args.name),args=args)
    if adata.shape[0] < args.max_num_cell:
        size_factor = adata.obs['size_factors'].values
        Zscore_data = preprocessing.scale(dataset)
        
        args.n_input = dataset.shape[1]
        print(args)
        init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, device=device)
        pretrain_model = train(init_model, Zscore_data, rawData, celltype, adj, r_adj, size_factor, device, args)
    
        asw, ari, nmi, pred_label, _, _ = clustering(pretrain_model, Zscore_data, rawData, celltype, adj, r_adj, size_factor, device, args)
        print("Final ASW %.3f, ARI %.3f, NMI %.3f"% (asw, ari, nmi))
        # output predicted labels
        # np.savetxt('./results/%s_predicted_label.csv'%(args.name),pred_label)
    #down-sampling input
    else:
        from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
        new_adata = utils.random_downsimpling(adata, args.max_num_cell)
        new_adj, new_r_adj = utils.adata_knn(new_adata, method = 'gauss', n_neighbors = args.n_neighbors, metric='cosine')
        new_Zscore_data = preprocessing.scale(new_adata.X)
        new_data = torch.Tensor(new_Zscore_data).to(device)
        new_adj = torch.Tensor(new_adj).to(device)
        size_factor = new_adata.obs['size_factors'].values
        Zscore_data = preprocessing.scale(dataset)
        data = torch.Tensor(Zscore_data).to(device)
        adj = torch.Tensor(adj).to(device)
        new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X
        new_celltype = new_adata.obs['celltype']
        args.n_input = dataset.shape[1]
        print(args)
        init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, device=device)
        pretrain_model = train(init_model, new_Zscore_data, new_rawData, new_celltype, new_adj, new_r_adj, size_factor, device, args)
        _, _, _, _, cluster_layer, model = clustering(pretrain_model, new_Zscore_data, new_rawData, 
                                                      new_celltype, new_adj, new_r_adj, size_factor, device, args)
    
        with torch.no_grad():
            z, _, _, _, _  = model(data,adj)
            _, p = train.loss_func(z, cluster_layer)
            pred_label = utils.dist_2_label(p)
            asw = np.round(silhouette_score(z.detach().cpu().numpy(), pred_label), 3)
            nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
            ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
            print("Final ASW %.3f, ARI %.3f, NMI %.3f"% (asw, ari, nmi))
        
        # np.savetxt('./results/%s_predicted_label.csv'%(args.name),pred_label)
