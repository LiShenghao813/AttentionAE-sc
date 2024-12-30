# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:57:57 2022

@author: LSH
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
import utils
from loss import ZINBLoss
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
import time
import random 

random.seed(1)

def train(init_model, Zscore_data, rawData, adj, r_adj, size_factor, device, args):
    start_time = time.time()
    
    start_mem = torch.cuda.max_memory_allocated(device=device)
    
    init_model.to(device)
    data = torch.Tensor(Zscore_data).to(device)
    sf = torch.autograd.Variable((torch.from_numpy(size_factor[:,None]).type(torch.FloatTensor)).to(device),
                           requires_grad=True)
    optimizer = torch.optim.Adam(init_model.parameters(), lr=args.lr)
    
    if type(adj) ==scipy.sparse._csr.csr_matrix:
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj).to(device)
        r_adj = torch.Tensor(r_adj.toarray()).to(device)
    else:
        adj = torch.Tensor(adj).to(device)
        r_adj = torch.Tensor(r_adj).to(device)
        
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5, last_epoch=-1)
    best_model = init_model
    loss_update = 100000
    for epoch in range(args.training_epoch):
        z, A_pred, pi, mean, disp = init_model(data, adj)
        l = ZINBLoss(theta_shape=(args.n_input,))
        zinb_loss = l(mean * sf, pi, target=torch.tensor(rawData).to(device), theta=disp)
        re_graphloss = torch.nn.functional.mse_loss(A_pred.view(-1), r_adj.view(-1))
        loss = zinb_loss + 0.1*re_graphloss
        
        if (epoch+1) % 10   == 0:
            print("epoch %d, loss %.4f, zinb_loss %.4f, re_graphloss %.4f" 
                            % (epoch+1, loss, zinb_loss, re_graphloss))
            
        if loss_update > loss:
            loss_update = loss
            best_model = init_model
            epoch_update = epoch
            
        if ((epoch - epoch_update) > 50):
            print("Early stopping at epoch {}".format(epoch_update))
            elapsed_time = time.time() - start_time
            max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
            print("Finish Training! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time, max_mem / 1024 / 1024))
            return best_model 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(init_model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
        scheduler.step()
    elapsed_time = time.time() - start_time
    max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
    print("Finish Training! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time, max_mem / 1024 / 1024))
    return best_model, elapsed_time
    
alpha = 1
def loss_func(z, cluster_layer):
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - cluster_layer) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()

    log_q = torch.log(q)
    loss = torch.nn.functional.kl_div(log_q, p, reduction='batchmean')
    return loss, p
   
def clustering(pretrain_model, Zscore_data, rawData, celltype, adj, r_adj, size_factor, device, args):
    start_time = time.time()
    
    start_mem = torch.cuda.max_memory_allocated(device=device)
    
    data = torch.Tensor(Zscore_data).to(device)
    if type(adj) ==scipy.sparse._csr.csr_matrix:
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj).to(device)
        r_adj = torch.Tensor(r_adj.toarray()).to(device)
    else:
        adj = torch.Tensor(adj).to(device)
        r_adj = torch.Tensor(r_adj).to(device)
        
    model = pretrain_model.to(device)
    sf = torch.autograd.Variable((torch.from_numpy(size_factor[:,None]).type(torch.FloatTensor)).to(device),
                          requires_grad=True)
    #cluster center
    with torch.no_grad():
        z, _, _, _, _  = model(data,adj)
        
    cluster_centers, init_label = utils.use_Leiden(z.detach().cpu().numpy(), resolution=args.resolution)
    cluster_layer = torch.autograd.Variable((torch.from_numpy(cluster_centers).type(torch.FloatTensor)).to(device),
                           requires_grad=True)
    asw = np.round(silhouette_score(z.detach().cpu().numpy(), init_label), 3)
    if celltype is not None:
        nmi = np.round(normalized_mutual_info_score(celltype, init_label), 3)
        ari = np.round(adjusted_rand_score(celltype, init_label), 3)
        print('init: ASW= %.3f, ARI= %.3f, NMI= %.3f' % (asw, ari, nmi)) 
    else:
        print('init: ASW= %.3f' % (asw)) 
        
    optimizer = torch.optim.Adam(list(model.enc_1.parameters()) + list(model.enc_2.parameters()) + 
                                  list(model.attn1.parameters()) + list(model.attn2.parameters()) + 
                                  list(model.gnn_1.parameters()) + list(model.gnn_2.parameters()) +
                                 list(model.z_layer.parameters()) + [cluster_layer], lr=0.001)   
    
    for epoch in range(args.clustering_epoch):
        z, A_pred, pi, mean, disp = model(data, adj)
        kl_loss, ae_p = loss_func(z, cluster_layer)
        l = ZINBLoss(theta_shape=(args.n_input,))
        zinb_loss = l(mean * sf, pi, target=torch.tensor(rawData).to(device), theta=disp)
        re_graphloss = torch.nn.functional.mse_loss(A_pred.view(-1), r_adj.to(device).view(-1))
        loss = kl_loss + 0.1 * zinb_loss + 0.01*re_graphloss
        loss.requires_grad_(True)
        label = utils.dist_2_label(ae_p)
        
        asw = silhouette_score(z.detach().cpu().numpy(), label)
        db = davies_bouldin_score(z.detach().cpu().numpy(), label)
        # ari = adjusted_rand_score(celltype, label)
        # nmi = normalized_mutual_info_score(celltype, label)
       

        if (epoch+1) % 10 == 0:
            print("epoch %d, loss %.4f, kl_loss %.4f, ASW %.3f"% (epoch+1, loss, kl_loss, asw))
        num = data.shape[0]
        tol=1e-3
        if epoch == 0:
            last_label = label
        else:
            delta_label = np.sum(label != last_label).astype(np.float32) / num
            last_label = label
            if epoch>20 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                elapsed_time = time.time() - start_time
                max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
                print("Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time, max_mem / 1024 / 1024))
                break  
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
    elapsed_time = time.time() - start_time
    max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
    print("Finish Clustering! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time, max_mem / 1024 / 1024))

    return [asw,db], last_label, cluster_layer, model, elapsed_time

