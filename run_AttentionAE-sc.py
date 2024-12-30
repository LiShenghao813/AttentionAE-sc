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
from train import train, clustering, loss_func
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001,help='learning rate, default:1e-3')
    parser.add_argument('--n_z', type=int, default=16, 
                        help='the number of dimension of latent vectors for each cell, default:16')
    parser.add_argument('--n_heads', type=int, default=8, 
                        help='the number of dattention heads, default:8')
    parser.add_argument('--n_hvg', type=int, default=2500, 
                        help='the number of the highly variable genes, default:2500')
    parser.add_argument('--training_epoch', type=int, default=200,
                        help='epoch of train stage, default:200')
    parser.add_argument('--clustering_epoch', type=int, default=100,
                        help='epoch of clustering stage, default:100')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='''the resolution of Leiden. The smaller the settings to get the more clusters
                        , advised to 0.1-1.0, default:1.0''')
    parser.add_argument('--connectivity_methods', type=str, default='gauss',
                        help='method for constructing the cell connectivity ("gauss" or "umap"), default:gauss')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='''The size of local neighborhood (in terms of number of neighboring data points) used 
                        for manifold approximation. Larger values result in more global views of the manifold, while 
                        smaller values result in more local data being preserved. In general values should be in the 
                        range 2 to 100. default:15''')
    parser.add_argument('--knn', type=int, default=False,
                        help='''If True, use a hard threshold to restrict the number of neighbors to n_neighbors, 
                        that is, consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights to
                        neighbors more distant than the n_neighbors nearest neighbor. default:False''')
    parser.add_argument('--name', type=str, default='Quake_10x_Spleen',
                        help='name of input file(a h5ad file: Contains the raw count matrix "X")')
    parser.add_argument('--celltype', type=str, default='known',
                        help='the true labels of datasets are placed in adata.obs["celltype"]')
    parser.add_argument('--save_pred_label', type=str, default=False,
                        help='To choose whether saves the pred_label to the dict "./pred_label"')
    parser.add_argument('--save_model_para', type=str, default=False,
                        help='To choose whether saves the model parameters to the dict "./model_save"')
    parser.add_argument('--save_embedding', type=str, default=False,
                        help='To choose whether saves the cell embedding to the dict "./embedding"')
    parser.add_argument('--save_umap', type=str, default=False,
                        help='To choose whether saves the visualization to the dict "./umap_figure"')
    parser.add_argument('--max_num_cell', type=int, default=4000,
                        help='''a maximum threshold about the number of cells use in the model building, 
                        4,000 is the maximum cells that a GPU owning 8 GB memory can handle. 
                        More cells will bemploy the down-sampling straegy, 
                        which has been shown to be equally effective,
                        but it's recommended to process data with less than 24,000 cells at a time''')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='use GPU, or else use cpu (setting as "False")')
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    simplefilter(action='ignore', category=FutureWarning)
    
    random.seed(1)
    if args.save_umap is True:
        umap_save_path = ['./umap_figure/%s_pred_label.png'%(args.name),'./umap_figure/%s_true_label.png'%(args.name)]
    else:
        umap_save_path = [None, None]
        
    adata, rawData, dataset, adj, r_adj = utils.load_data('./Data/AnnData/{}'.format(args.name),args=args)
    
    if args.celltype == "known":  
        celltype = adata.obs['celltype'].tolist()
    else:
        celltype = None
        
    if adata.shape[0] < args.max_num_cell:
        size_factor = adata.obs['size_factors'].values
        Zscore_data = preprocessing.scale(dataset)
        
        args.n_input = dataset.shape[1]
        print(args)
        init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads, device=device)
        pretrain_model, _ = train(init_model, Zscore_data, rawData, adj, r_adj, size_factor, device, args)
        metric, pred_label, _, model, _ = clustering(pretrain_model, Zscore_data, rawData, celltype, 
                                                  adj, r_adj, size_factor, device, args)
        asw = metric[0]
        db  = metric[1]
        if celltype is not None:
            nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
            ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
            print("Final ASW %.3f, DB %.3f, ARI %.3f, NMI %.3f"% (asw, db, ari, nmi))
            
        else:
            print("Final ASW %.3f, DB %.3f"% (asw, db))
        data = torch.Tensor(Zscore_data).to(device)
        if type(adj) == scipy.sparse._csr.csr_matrix:
            adj = utils.sparse_mx_to_torch_sparse_tnsor(adj).to(device)
        else:
            adj = torch.Tensor(adj).cpu()
        with torch.no_grad():
            z, _, _, _, _  = model(data,adj)
            if args.save_umap is True:
                utils.umap_visual(z.detach().cpu().numpy(), 
                                  label = pred_label, 
                                  title='predicted label', 
                                  save_path = umap_save_path[0],
                                  asw_used=True)
                if args.celltype == "known":  
                    utils.umap_visual(z.detach().cpu().numpy(), 
                                      label = celltype, 
                                      title='true label', 
                                      save_path = umap_save_path[1])
        if args.save_embedding is True:
            np.savetxt('./embedding/%s.csv'%(args.name), z.detach().cpu().numpy())
        if args.save_pred_label is True:
            np.savetxt('./pred_label/%s.csv'%(args.name),pred_label)
        if args.save_model_para is True:
            torch.save(model.state_dict(), './model_save/%s.pkl'%(args.name))
        
        # output predicted labels
        # np.savetxt('./results/%s_predicted_label.csv'%(args.name),pred_label)

    #down-sampling input
    else:
        new_adata = utils.random_downsimpling(adata, args.max_num_cell)
        new_adj, new_r_adj = utils.adata_knn(new_adata, method = args.connectivity_methods, 
                                             knn=args.knn, n_neighbors=args.n_neighbors)
        try: 
            new_Zscore_data = preprocessing.scale(new_adata.X.toarray())
            new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X.toarray()
        except:
            new_Zscore_data = preprocessing.scale(new_adata.X)
            new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X
            
        
        
        size_factor = new_adata.obs['size_factors'].values
        try: 
            Zscore_data = preprocessing.scale(dataset.toarray())
            
        except:
            Zscore_data = preprocessing.scale(dataset)
            
        
        new_celltype = new_adata.obs['celltype']
        args.n_input = dataset.shape[1]
        print(args)
        init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads,device=device)
        pretrain_model, train_elapsed_time  = train(init_model, new_Zscore_data, new_rawData,
                                                    new_adj, new_r_adj, size_factor, device, args)
        _, _, cluster_layer, model, _ = clustering(pretrain_model, new_Zscore_data, new_rawData, 
                                                   new_celltype, new_adj, new_r_adj, size_factor, device, args)
        
        copy_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads, device=torch.device('cpu'))
        copy_model.load_state_dict(model.state_dict())
        
        data = torch.Tensor(Zscore_data).cpu()
        if type(adj) == scipy.sparse._csr.csr_matrix:
            adj = utils.sparse_mx_to_torch_sparse_tnsor(adj).to(device)
        else:
            adj = torch.Tensor(adj).cpu()
            
        with torch.no_grad():
            z, _, _, _, _  = copy_model(data,adj)
            _, p = loss_func(z, cluster_layer.cpu())
            pred_label = utils.dist_2_label(p)
            
            if args.celltype == "known":  
                asw = np.round(silhouette_score(z.detach().cpu().numpy(), pred_label), 3)
                db = np.round(davies_bouldin_score(z.detach().cpu().numpy(), pred_label), 3)
                nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
                ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
                print("Final ASW %.3f, DB %.3f, ARI %.3f, NMI %.3f"% (asw, db, ari, nmi))
            else:
                asw = np.round(silhouette_score(z.detach().cpu().numpy(), pred_label), 3)
                db = np.round(davies_bouldin_score(z.detach().cpu().numpy(), pred_label), 3)
                print("Final ASW %.3f, DB %.3f"% (asw,db))
                
            if args.save_umap is True:
                utils.umap_visual(z.detach().cpu().numpy(), 
                                  label = pred_label, 
                                  title='predicted label', 
                                  save_path = umap_save_path[0],
                                  asw_used=True)
                if args.celltype == "known":  
                    utils.umap_visual(z.detach().cpu().numpy(), 
                                      label = celltype, 
                                      title='true label', 
                                      save_path = umap_save_path[1])
        if args.save_embedding is True:
            np.savetxt('./embedding/%s.csv'%(args.name), z.detach().cpu().numpy())
        if args.save_pred_label is True:
            np.savetxt('./pred_label/%s.csv'%(args.name), pred_label)
        if args.save_model_para is True:
            torch.save(model.state_dict(), './model_save/%s.pkl'%(args.name))
